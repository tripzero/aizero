from datetime import datetime
from resource import HOURS, MINS, Resource

from resource_py3 import Py3Resource as resource_poll

class DayOfWeekResource(Resource):
    def __init__(self):
        Resource.__init__(self, "DayOfWeekResource", ["day_of_week"])
        self.poller = resource_poll(self.poll_func, MINS(10))


    def getValue(self, pn):
        return datetime.now().date().weekday()


    def poll_func(self):
        try:
            self.setValue("day_of_week", datetime.now().date().weekday())
        except:
            print("failed to set day of week resource")


class HourOfDayResource(Resource):
    def __init__(self, update_rate = MINS(10)):
        Resource.__init__(self, "HourOfDayResource", ["hour_of_day"])
        self.poller = resource_poll(self.poll_func, update_rate)
        self.override_value = False


    def setValue(self, pn, value):
        if pn == "hour_of_day":
            self.override_value = True

            try:
                """see if value is a datetime-compitible object"""
                d = value.date()
                h = value.time()

                value = HourOfDayResource.hour(value)
            except:
                pass

        super().setValue(pn, value)

    def getValue(self, pn):
        if self.override_value:
            return super().getValue(pn)

        return HourOfDayResource.hour(datetime.now())


    @staticmethod
    def hour(date_time):
        return date_time.time().hour + round(date_time.time().minute / 60.0, 1)


    def poll_func(self):
        try:
            self.setValue("hour_of_day", self.getValue(None))
        except:
            print("failed to set hour of day resource")


def test_day_of_week():

    dow = DayOfWeekResource()

    weekday = datetime.now().date().weekday()

    assert weekday == dow.getValue("day_of_week")


def test_hour_of_day():

    hod = HourOfDayResource()

    hour = datetime.now().time().hour + round(datetime.now().time().minute / 60, 1)

    assert hour == hod.getValue("hour_of_day")
    assert hour == HourOfDayResource.hour(datetime.now())
