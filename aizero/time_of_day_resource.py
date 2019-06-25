from datetime import datetime
from numpy import float64

from aizero.resource import MINS, Resource
from aizero.resource_py3 import Py3Resource as resource_poll
from .sys_time import get_current_datetime


class DayOfWeekResource(Resource):
    def __init__(self):
        super().__init__("DayOfWeekResource", ["day_of_week"])
        self.poller = resource_poll(self.poll_func, MINS(10))

        self.poll_func()

    def get_value(self, pn):
        return float64(get_current_datetime().date().weekday())

    def set_value(self, pn, value):

        if pn == "day_of_week":
            try:
                value.date()

                value = value.date().weekday()
            except Exception:
                pass

        super().set_value(pn, value)

    def poll_func(self):
        try:
            #print("debug: trying to set day_of_week")
            self.set_value("day_of_week", self.get_value("day_of_week"))
            #print("day_of_week dataframe: {}".format(self.dataframe))

        except Exception as ex:
            print("failed to set day of week resource: {}".format(ex))


class HourOfDayResource(Resource):
    def __init__(self, update_rate=MINS(1)):
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
            except Exception:
                pass

        super().set_value(pn, value)

    def get_value(self, pn):
        if self.override_value:
            return super().get_value(pn)

        return HourOfDayResource.hour(get_current_datetime())

    @staticmethod
    def hour(date_time):
        return date_time.time().hour + round(date_time.time().minute / 60.0, 1)

    def poll_func(self):
        try:
            hod = HourOfDayResource.hour(get_current_datetime())

            # print("debug: trying to set hour_of_day to {}".format(
            #    hod))
            self.set_value("hour_of_day", hod)
            # print("hour_of_day dataframe ({}): \n{}".format(
            #    len(self.dataframe.index),
            #    self.dataframe))

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

    weekday = get_current_datetime().date().weekday()

    assert weekday == dow.getValue("day_of_week")


def test_hour_of_day():
    Resource.clearResources()

    hod = HourOfDayResource()

    hour = get_current_datetime().time().hour + \
        round(get_current_datetime().time().minute / 60, 1)

    assert hour == hod.getValue("hour_of_day")
    assert hour == HourOfDayResource.hour(get_current_datetime())


def test_dataframe_has_timestamp_column():
    Resource.clearResources()
    dow = DayOfWeekResource()
    hod = HourOfDayResource()

    assert "timestamp" in dow.dataframe.columns
    assert "timestamp" in hod.dataframe.columns


def test_multiple_rows_in_dataframe():
    Resource.clearResources()

    dow = DayOfWeekResource()
    hod = HourOfDayResource()

    dow.set_value("day_of_week", 1)
    dow.set_value("day_of_week", 2)
    dow.set_value("day_of_week", 3)
    dow.set_value("day_of_week", 4)

    assert len(dow.dataframe) == 5

    hod.set_value("hour_of_day", 1.2)
    hod.set_value("hour_of_day", 2.2)
    hod.set_value("hour_of_day", 1.5)
    hod.set_value("hour_of_day", 13.1)

    assert len(hod.dataframe) == 5
