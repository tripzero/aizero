""" Night time resource
    Uses "latitude" and "longitude" from ConfigurationResource
"""

from aizero.resource import Resource, MINS, ResourceNotFoundException
from aizero.resource import get_resource
from aizero.resource_py3 import Py3Resource as resource_poll

from datetime import datetime, time
from astral import Location


class NightTime(Resource):

    isNightTime = False

    def __init__(self, name="NightTime", lat=None,
                 lon=None, timezone="US/Pacific"):
        Resource.__init__(
            self, name, ["nightTime", "night_time", "sunset", "sunrise"])

        if lat is None or lon is None:
            try:
                config = get_resource("ConfigurationResource")
                lat = config.get_value("latitude")
                lon = config.get_value("longitude")

            except ResourceNotFoundException:
                raise Exception(
                    "NightTime requires lat/lon set or ConfigurationResource")

            if lat is None or lon is None:
                raise Exception(
                    "NightTime: missing latitude/longitude in ConfigurationResource")

        print("using lat/lon: {}, {}".format(lat, lon))

        self.location = Location()
        self.location.latitude = float(lat)
        self.location.longitude = float(lon)
        self.location.timezone = timezone

        self.process()

        self.poller = resource_poll(self.process, MINS(1))

    def is_day(self):
        return not self.isNightTime

    @property
    def is_night(self):
        return self.isNightTime

    def process(self):
        t = datetime.now().time()

        try:
            current_time_resource = Resource.resource("CurrentTimeResource")
            has_time = current_time_resource.getValue("datetime")
            if has_time is not None:
                t = has_time.time()
        except ResourceNotFoundException:
            pass
        except Exception as e:
            print("night_time: another bad exception: {}".format(e))

        self.isNightTime = ((t > self.location.sunset().time() and
                             t <= time(23, 59, 59)) or
                            (t >= time(0) and
                             t < self.location.sunrise().time()))

        self.sunset = self.location.sunset().time()
        self.sunrise = self.location.sunrise().time()

        self.setValue("nightTime", self.isNightTime)
        self.setValue("night_time", self.isNightTime)
        self.setValue("sunset", str(self.location.sunset().time()))
        self.setValue("sunrise", str(self.location.sunrise().time()))


def test_use_current_time_resource():
    current_time_resource = Resource("CurrentTimeResource", ["datetime"])

    current_time_resource.setValue("datetime", datetime(2017, 1, 1, 12))

    night_time = NightTime()

    night_time.process()

    assert night_time.is_day()

    current_time_resource.setValue("datetime", datetime(2017, 1, 1, 23))

    night_time.process()

    assert not night_time.is_day()
