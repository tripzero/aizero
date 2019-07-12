import asyncio
import json
import math
import numpy as np
import os.path
import time as tm

from pysolar import radiation
from pysolar import solar
from datetime import datetime, timedelta

from aizero.resource import Resource, get_resource, MINS
from aizero.sys_time import get_current_datetime


def get_solar_radiation(lat, lon, date):
    """
            date should be in UTC
    """

    solarAlt = solar.get_altitude(lat, lon, date)

    if solarAlt <= 0:
        return 0

    return radiation.get_radiation_direct(date, solarAlt)


def get_solar_radiation_data(date, days, lat, lon):
    radiation = []

    for i in range(days):
        daily_radiation = 0.0
        counted_hrs = 0.0
        for i in range(3600 * 24):

            r = get_solar_radiation(lat, lon, date)

            if r:
                radiation.append(r)

            date += timedelta(seconds=1)

    return radiation


class SolarInsolation(Resource):

    def __init__(self):
        super().__init__("SolarInsolation", variables=["current_power"])

        self.lat = get_resource("ConfigurationResource").config["latitude"]
        self.lon = get_resource("ConfigurationResource").config["longitude"]

    def get_solar_radiation(self, date):
        return get_solar_radiation(self.lat, self.lon, date)

    @asyncio.coroutine
    def poll(self):
        while True:
            try:
                t = get_current_datetime(utc=True)
                self.set_value("current_power", get_solar_radiation(self.lat,
                                                                    self.lon,
                                                                    t))
            except Exception as ex:
                print("error getting solar insolation: {}".format(ex))

            yield from asyncio.sleep(MINS(3))


if __name__ == "__main__":

    import argparse

    def valid_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            msg = "Not a valid date: '{}'".format(s)
            raise argparse.ArgumentTypeError(msg)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--greenhouse", help="greenhouse characteristics file", default=None, type=str)
    parser.add_argument("--lat", help="latitude", default=None, type=float)
    parser.add_argument("--lon", help="longitude", default=None, type=float)
    parser.add_argument(
        "--start", help="starting date in format YYYY-MM-DD", default=datetime, type=valid_date)
    parser.add_argument("--days", help="number of days", default=30, type=int)
    args = parser.parse_args()

    greenhouse = {}

    if args.greenhouse:
        with open(args.greenhouse, "r") as configFile:
            greenhouse = json.loads(configFile.read())

    if not args.lat and "latitude" in greenhouse:
        args.lat = greenhouse["latitude"]

    if not args.lon and "longitude" in greenhouse:
        args.lon = greenhouse['longitude']

    radiation = get_solar_radiation_data(
        args.start, args.days, args.lat, args.lon)

    print("avg radiation: {}W/m^2".format(np.average(radiation)))
    print("max radiation: {}W/m^2".format(np.max(radiation)))
    print("min radiation: {}W/m^2".format(np.min(radiation)))
