from datetime import datetime
from numpy import float64

from aizero.resource import MINS, Resource
from aizero.resource_py3 import Py3Resource as resource_poll


class DayOfWeekResource(Resource):
    def __init__(self):
        super().__init__("DayOfWeekResource", ["day_of_week"])
        self.poller = resource_poll(self.poll_func, MINS(10))

        self.poll_func()

    def get_value(self, pn):
        return float64(datetime.now().date().weekday())

    def poll_func(self):
        try:
            print("debug: trying to set day_of_week")
            self.set_value("day_of_week", self.get_value("day_of_week"))
            print("day_of_week dataframe: {}".format(self.dataframe))

        except Exception as ex:
            print("failed to set day of week resource: {}".format(ex))


class HourOfDayResource(Resource):
    def __init__(self, update_rate=MINS(10)):
        super().__init__("HourOfDayResource", ["hour_of_day"])
        self.poller = resource_poll(self.poll_func, update_rate)
        self.override_value = False

        self.poll_func()

    def set_value(self, pn, value):
        if pn == "hour_of_day":
            self.override_value = True

            try:
                """see if value is a datetime-compitible object"""
                value.date()
                value.time()

                value = HourOfDayResource.hour(value)
            except:
                pass

        super().set_value(pn, value)

    def get_value(self, pn):
        if self.override_value:
            return super().get_value(pn)

        return HourOfDayResource.hour(datetime.now())

    @staticmethod
    def hour(date_time):
        return date_time.time().hour + round(date_time.time().minute / 60.0, 1)

    def poll_func(self):
        try:
            self.set_value("hour_of_day", self.get_value("hour_of_day"))
        except Exception as ex:
            import sys
            import traceback
            print("failed to set hour of day resource {}".format(ex))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)


def test_day_of_week():
    Resource.clearResources()

    dow = DayOfWeekResource()

    weekday = datetime.now().date().weekday()

    assert weekday == dow.getValue("day_of_week")


def test_hour_of_day():
    Resource.clearResources()

    hod = HourOfDayResource()

    hour = datetime.now().time().hour + round(datetime.now().time().minute / 60, 1)

    assert hour == hod.getValue("hour_of_day")
    assert hour == HourOfDayResource.hour(datetime.now())


def test_dataframe_has_timestamp_column():
    Resource.clearResources()
    dow = DayOfWeekResource()
    hod = HourOfDayResource()

    assert "timestamp" in dow.dataframe.columns
    assert "timestamp" in hod.dataframe.columns
