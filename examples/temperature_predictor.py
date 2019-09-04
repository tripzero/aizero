"""
temperature predictor.

this examples uses a config with the following keys:

    {
        "latitude" : 42.152343,
        "longitude" : 122.234343,
        "weather_key" : "youropenweathermapapikey",
        "ecobee_apikey" : "yourecobeeapikey",
        "ecobee_thermostat_name" : "Thermostatname",
        "solaredge_api" : "solaredge_api"
    }
"""

from aizero.resource import Resource, ResourceRequires
from aizero.learningresource_keras import *


import asyncio
import pytz

from tensorflow import keras

from aizero.learningresource_keras import *
from aizero.resource import Resource, MINS, HOURS
from aizero.resource import get_resource as gr
from aizero.sys_time import get_current_datetime


class TemperaturePredictor(Resource):
    """
    Predicts the indoor temperature of a building or room
    """

    def __init__(self, temperature_resource,
                 additional_features=None):
        """
        temperature_resource: temperature resource to predict, usually from
                              a thermostat sensor
        """
        name = "TemperaturePredictor_{}".format(temperature1)
        super().__init__(name, variables=[
            "temperature",
            "solar_power"
        ])

        self.temp1 = temperature_resource

        self.solar = solar_power
        self.additional_features = additional_features

        self.predictor = None
        self.solar_predictor = None

        self.rsrcs = ResourceRequires([
            temperature_resource,
            "SolarPower",
            "weather",
            "SolarInsolation"], self.init_predictor)

    def init_predictor(self):

        temp1 = FeatureColumn("temperature1", self.rsrcs(self.temp1),
                              "temperature")
        temp2 = FeatureColumn("weather_temperature", self.rsrcs("weather"),
                              "temperature")

        solar = FeatureColumn("solar_power", self.rsrcs(self.solar),
                              "current_power")

        weather = FeatureColumn("weather_cloud_cover",
                                self.rsrcs("weather"),
                                "cloud_cover")

        solar_insolation = FeatureColumn("solar_insolation",
                                         self.rsrcs("SolarInsolation"),
                                         "current_power")

        all_features = [temp1, temp2, solar, weather,
                        solar_insolation]

        if self.additional_features is not None:
            all_features.extend(self.additional_features)

        self.predictor = Learning(model_subdir="temperature_predictor_2",
                                  features=all_features,
                                  prediction_feature="temperature1",
                                  persist=True)

        solar_features = [
            solar, weather, solar_insolation]

        self.solar_predictor = Learning(model_subdir="solar_predictor",
                                        features=solar_features,
                                        prediction_feature="solar_power",
                                        persist=True)

        asyncio.get_event_loop().create_task(self.train_loop())

    def predict_solar(self, t=None):

        if t is None:
            t = get_current_datetime()

        weather = gr("weather")

        forecast = weather.get_standard_forecast(t)
        cloud_cover = forecast["forecast_cloud_cover"]

        solar_insolation = gr("SolarInsolation")

        radiation_forecast = solar_insolation.get_solar_radiation(
            t.astimezone(pytz.utc))

        return self.solar_predictor.predict(replace_features=[
            ReplaceFeatureColumn("solar_insolation", radiation_forecast),
            ReplaceFeatureColumn("weather", forecast)
        ])

    def predict_temperature(self, t=None):
        if t is None:
            t = get_current_datetime()

        weather = gr("weather")

        forecast = weather.get_forecast(t)

        cloud_cover = forecast['forecast_cloud_cover']
        weather_temperature = forecast['forecast_high']

        solar_insolation = gr("SolarInsolation")

        radiation_forecast = solar_insolation.get_solar_radiation(
            t.astimezone(pytz.utc))

        solar_power_prediction = self.predict_solar(t)

        return self.solar_predictor.predict(replace_features=[
            ReplaceFeatureColumn("solar_insolation", radiation_forecast),
            ReplaceFeatureColumn("weather_cloud_cover", cloud_cover),
            ReplaceFeatureColumn("weather_temperature", weather_temperature),
            ReplaceFeatureColumn("solar_power", solar_power_prediction)
        ])

    @asyncio.coroutine
    def train_loop(self):

        yield from asyncio.sleep(5)

        try:
            print("training temperature predictor")

            hist = self.predictor.train(epochs=1000)

            print("history: \n{}".format(pd.DataFrame(hist.history).tail()))

            self.predictor.plot_history(hist)
            # self.predictor.plot()

            print("training solar predictor")

            hist = self.solar_predictor.train(epochs=1000)
            print("history: \n{}".format(pd.DataFrame(hist.history).tail()))

        except Exception as ex:
            print("failed initial training. maybe no data yet?")
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

        while True:
            yield from asyncio.sleep(HOURS(12))

            try:
                self.predictor.train()
                self.solar_predictor.train()

            except Exception as ex:
                print("failed training. maybe no data yet?")
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=6, file=sys.stdout)

    @asyncio.coroutine
    def poll(self):
        while True:
            try:
                if self.predictor is None:
                    yield from asyncio.sleep(MINS(5))
                    continue

                predicted_temperature = self.predictor.predict()

                print("predicted temperature: {}".format(
                    predicted_temperature))
                print("actual temperature: {}".format(
                    gr(self.temp1).get_value("temperature")))

                self.set_value("temperature", predicted_temperature)

                predicted_solar = self.predict_solar()

                print("predicted solar power: {}".format(
                    predicted_solar))
                print("actual solar power: {}".format(
                    gr(self.solar).get_value("current_power")))

                self.set_value("solar_power", predicted_solar)

            except AttributeError as ae:
                print("{} - Error. don't have a predictor yet? {}".format(
                    self.name, ae))
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=12, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=12, file=sys.stdout)
                pass

            except Exception as ex:
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=6, file=sys.stdout)

            yield from asyncio.sleep(MINS(5))
            # yield from asyncio.sleep(MINS(15))


def main():
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    import argparse
    import json

    from aizero.configuration_resource import ConfigurationResource
    from aizero.ecobee_resource import EcobeeResource
    from aizero.solaredge import SolarPower
    from aizero.solar_insolation import SolarInsolation
    from aizero.openweathermap_resource import WeatherResource

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest="configPath",
                        help="specify config.", default="config.json")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    # check config syntax...
    with open(args.configPath, "r") as configFile:
        json.loads(configFile.read())

    ConfigurationResource(args.configPath)

    SolarInsolation()
    solar = SolarPower()
    weather = WeatherResource()
    ecobee = EcobeeResource()

    tp1 = TemperaturePredictor(ecobee.name)

    loop.run_forever()


if __name__ == "__main__":

    main()
