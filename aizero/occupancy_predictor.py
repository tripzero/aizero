import asyncio
import sys
import traceback

from numpy import float64

from aizero.device_resource import DeviceResource, RunIfCanPolicy
from aizero.resource import Resource, MINS, HOURS, get_resource
from aizero.learningresource_keras import *
from aizero.time_of_day_resource import HourOfDayResource, DayOfWeekResource
from aizero.sys_time import get_current_datetime
from aizero.time_of_use_resource import Modes
from aizero.resource_py3 import Py3Resource as resource_poll


class OccupancyPredictorResource(DeviceResource):
    def __init__(self, name="OccupancyPredictorResource",
                 occupancy_resource="", prediction_threshold=0.70):
        super().__init__(name,
                         power_usage=100,
                         variables=["occupancy", "prediction"])
        # runtime_modes=[Modes.off_peak],

        self.did_train = False

        self.runtime_policy = RunIfCanPolicy(
            conditions=[lambda: self.did_train])

        self.prediction_threshold = prediction_threshold

        if occupancy_resource == "":
            occupancy_resource = "EcobeeResource"
        else:
            self.name = "{}_OccuppancyPredictorResource".format(
                occupancy_resource)

        self.occupancy_resource = occupancy_resource

        self.model_dir = "{}_occupancy_predictor".format(occupancy_resource)

        Resource.waitResource(
            [occupancy_resource], self.init_predictor)

    def init_predictor(self):
        print("initializing occupancy predictor")

        occupancy_feature = FeatureColumn("occupancy", get_resource(
            self.occupancy_resource), "occupancy")

        day_of_week_feature = FeatureColumn("day_of_week",
                                            DayOfWeekResource(),
                                            "day_of_week")
        hour_feature = FeatureColumn("hour_of_day",
                                     HourOfDayResource(),
                                     "hour_of_day")

        all_features = [occupancy_feature, day_of_week_feature, hour_feature]

        self.predictors = Learning(model_subdir=self.model_dir,
                                   features=all_features,
                                   prediction_feature="occupancy",
                                   persist=True)

        self.poller = resource_poll(self.poll_func, MINS(10))
        self.poller = resource_poll(self.wait_can_run, HOURS(1))

    def train(self, and_test=True):
        self.predictors.train(and_test=and_test)

    def predict_occupancy(self, date_to_predict=None):
        """ Return likelyhood that home has at least one human in it
            between 0-100"""
        try:
            if date_to_predict is None:
                date_to_predict = get_current_datetime()

            hour = HourOfDayResource.hour(date_to_predict)

            hour_of_week_fl = FakeFeatureColumn(
                "day_of_week", float64(date_to_predict.date().weekday()))

            hour_of_day_fl = FakeFeatureColumn("hour_of_day", hour)

            occupancy = self.predictors.predict(
                replace_features=[hour_of_day_fl,
                                  hour_of_week_fl])

            print("predicted occupancy: {}".format(round(occupancy, 2)))

            return occupancy

        except ValueError:
            print("Failed to predict. Model probably not trained yet")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

        except Exception:
            print("failed to predict occupancy")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

    def poll_func(self):
        predict_val = self.predict_occupancy()
        if predict_val is not None:
            self.setValue("occupancy", bool(
                predict_val > self.prediction_threshold))
            self.setValue('prediction', float(round(predict_val, 2)))

    def run(self):
        DeviceResource.run(self)

        self.did_train = True

        self.train()
        """except Exception as ex:
            print("Failed to train occupancy_predictor {} - {}".format(
                self.name, ex))
            raise(ex)
            self.did_train = False
            """

        self.stop()

    def stop(self):
        DeviceResource.stop(self)


def test_occupancy_predictor():

    from remoteresource import RemoteRestResource
    from hammock import Hammock

    class CR(Resource):
        def __init__(self):
            super().__init__("ConfigurationResource", [])
            self.config = {"db_root": "/home/kev/nas2/.cache"}

    config_resource = CR()

    thermostat = RemoteRestResource("EcobeeResource", Hammock(
        "https://192.168.1.40:8079/EcobeeResource/EcobeeResource"), poll_rate=3)
    weather = RemoteRestResource("weather", Hammock(
        "https://192.168.1.40:8079/weather/weather"), poll_rate=3)

    op = OccupancyPredictorResource()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(4))

    with open("24_occupancy_prediction.csv", "w") as f:

        d = datetime.now()
        for n in range(7):
            for i in range(240):
                d += timedelta(minutes=10)
                prediction = op.predict_occupancy(d)
                print("{} - {}".format(d, prediction))

                f.write('{}, {}, {}\n'.format(d.date().weekday(),
                                              HourOfDayResource.hour(d), prediction))


def test_occupancy_accuracy():

    from remoteresource import RemoteRestResource
    from hammock import Hammock
    import math
    import numpy as np

    class CR(Resource):
        def __init__(self):
            super().__init__("ConfigurationResource", [])
            db_root = "{}/.cache".format(os.environ["HOME"])
            self.config = {"db_root": db_root}

    config_resource = CR()

    thermostat = RemoteRestResource("EcobeeResource", Hammock(
        "https://192.168.1.40:8079/EcobeeResource/EcobeeResource"), poll_rate=3)
    weather = RemoteRestResource("weather", Hammock(
        "https://192.168.1.40:8079/weather/weather"), poll_rate=3)

    op = OccupancyPredictorResource()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(4))
    op.init_predictor()

    op.train()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(4))

    layers = op.predictors.all_layers

    num_values = len(layers[0].values)

    print("number of total values: {}".format(num_values))

    values_occupancy = layers[0].values
    values_day_of_week = layers[1].values
    values_hour_of_day = layers[2].values

    error_rate = []

    hour = 7.0

    with open("occupancy_prediction_accuracy.csv", "w") as f:
        for i in range(num_values):
            try:
                if values_hour_of_day[i] < hour or values_hour_of_day[i] > hour + 1:
                    continue

                d = datetime(
                    year=2019, month=12, day=values_day_of_week[i]+1, hour=math.floor(values_hour_of_day[i]))

                prediction = op.predict_occupancy(d)
                occupancy = 0.0
                if values_occupancy[i] == True:
                    occupancy = 1.0

                output_str = "{}, {}, {}, {}, {}\n".format(values_day_of_week[i],
                                                           values_hour_of_day[i],
                                                           occupancy,
                                                           round(
                    prediction, 2),
                    abs(occupancy - prediction))

                f.write(output_str)
                print(output_str)
                error_rate.append(abs(occupancy - prediction))

            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=6, file=sys.stdout)

    print("average error rate: {}".format(np.mean(error_rate)))


def main():
    # test_occupancy_predictor()
    test_occupancy_accuracy()


if __name__ == "__main__":
    main()
