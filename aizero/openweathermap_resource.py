import asyncio
import sys
import traceback

from hammock import Hammock
from datetime import datetime, timedelta

from aizero.resource import Resource, MINS, get_resource
from aizero.resource_py3 import Py3Resource as resource_poll
from aizero.sys_time import get_current_datetime
from aizero.utils import run_thread


def last_day_of_month(any_day):
    next_month = any_day.replace(
        day=28) + timedelta(days=4)  # this will never fail
    return (next_month - timedelta(days=next_month.day)).date().day


def weatherConditions(key, lat, lon):
    f = Hammock(
        'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric'.format(
            lat, lon, key))
    weather = f.GET().json()

    current_observation = {}
    current_observation["temperature"] = weather['main']['temp']
    current_observation["condition"] = weather['weather'][0]["main"]
    current_observation["humidity"] = weather['main']['humidity']
    current_observation["cloud_cover"] = weather['clouds']['all']
    return current_observation


def hourly(key, forecast_date, lat, lon):
    f = Hammock(
        'https://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}&units=metric'.format(
            lat, lon, key))

    hourly_forecast = f.GET().json()

    forecasts = hourly_forecast["list"]

    for forecast in forecasts:
        h = datetime.fromtimestamp(forecast["dt"])

        if h - forecast_date < timedelta(hours=3):
            return forecast


def hourly_temperature(key, forecast_date, lat, lon):
    """ return temperature for given city during the specified hour or None
        if no forecast for that hour
    """
    forecast = hourly(key, forecast_date, lat, lon)

    if forecast:
        return forecast["main"]["temp"]


def forecast(key, lat, lon):
    now = get_current_datetime()
    day_tomorrow = (now + timedelta(days=1))

    return hourly(key, day_tomorrow, lat, lon)


class WeatherResource(Resource):

    def __init__(self):
        Resource.__init__(self, "weather", ["temperature",
                                            "humidity",
                                            "forecast_high",
                                            "forecast_low",
                                            "forecast_conditions",
                                            "condition",
                                            "current_observation",
                                            "cloud_cover",
                                            "forecast_cloud_cover"])
        self.key = get_resource(
            "ConfigurationResource").config["weather_key"]
        self.lat = get_resource(
            "ConfigurationResource").config["latitude"]
        self.lon = get_resource(
            "ConfigurationResource").config["longitude"]

        self.poller = resource_poll(
            self.poll_func, MINS(15), is_coroutine=True)

    def get_hourly_temperature(self, hour):
        return hourly_temperature(self.key, hour, "Hillsboro")

    def get_forecast(self, forecast_date=None):
        if forecast_date is None:
            forecast_date = get_current_datetime()

        return hourly(self.key, forecast_date, self.lat, self.lon)

    def get_standard_forecast(self, forecast_date=None):
        if forecast_date is None:
            forecast_date = get_current_datetime()

        fc = hourly(self.key, forecast_date, self.lat, self.lon)

        fc_standard = {}

        fc_standard["forecast_high"] = fc["main"]["temp_max"]
        fc_standard["forecast_low"] = fc["main"]["temp_min"]
        fc_standard["forecast_cloud_cover"] = fc['clouds']['all']
        fc_standard['forecast_conditions'] = fc["weather"][0]["main"]

    @asyncio.coroutine
    def poll_func(self):

        try:
            current_observation = yield from run_thread(weatherConditions,
                                                        self.key,
                                                        self.lat,
                                                        self.lon)
            if current_observation:
                self.setValue(
                    "temperature", current_observation["temperature"])
                self.setValue("humidity", current_observation["humidity"])
                self.setValue("condition", current_observation["condition"])
                self.set_value("cloud_cover",
                               current_observation["cloud_cover"])

            tomorrow = yield from run_thread(forecast, self.key,
                                             self.lat, self.lon)

            if tomorrow is not None:
                self.setValue("forecast_high", tomorrow["main"]["temp_max"])
                self.setValue("forecast_low", tomorrow["main"]['temp_min'])
                self.setValue("forecast_conditions",
                              tomorrow["weather"][0]["main"])
                self.set_value("forecast_cloud_cover",
                               tomorrow['clouds']['all'])

        except Exception:
            print("error FAILED getting weather")

            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)


if __name__ == "__main__":
    api_key = sys.argv[1]
    lat = sys.argv[2]
    lon = sys.argv[3]

    tomorrow = forecast(api_key, lat, lon)

    print("tomorrow: {}".format(tomorrow))

    print("tomorrow's low: {}".format(tomorrow["main"]["temp_min"]))
    print("tomorrow's high: {}".format(tomorrow["main"]["temp_max"]))
    print("tomorrow's conditions: {}".format(tomorrow["weather"][0]["main"]))

    print("forecast for next hour: {}C".format(hourly_temperature(
        api_key, datetime.now()+timedelta(hours=4), lat, lon)))
