from aizero.resource import Resource, MINS
from aizero.resource import ResourceNotFoundException
""" time_of_use_resource.py
    Determines the current time of use mode.

    modes are :
    0 : off-peak
    1 : mid-peak
    2 : on-peak

"""

from datetime import datetime, date
import traceback
import sys

from aizero.resource_py3 import Py3Resource as resource_poll


def is_schedule(schedule, cur_date):

    schedule_months = schedule["months"]

    begin_year = cur_date.year

    begin_month = schedule_months[0]
    end_month = schedule_months[1]
    end_year = cur_date.year

    if end_month < begin_month:
        end_year += 1

        if cur_date.month <= end_month:
            begin_year -= 1

    begin_month = date(begin_year, schedule_months[0], 1)
    end_month = date(end_year, end_month, 30)

    # print("{} <= {} <= {}".format(begin_month, cur_date, end_month))

    return begin_month <= cur_date <= end_month


def is_mode(mode, cur_hour):

    # cur_hour =  24 - cur_hour

    is_the_mode = False

    for m in mode:
        m1 = m[0]
        m2 = m[1]

        is_the_mode |= m1 <= cur_hour <= m2
        is_the_mode |= m2 < m1 and (m1 <= cur_hour or cur_hour <= m2)

    return is_the_mode


class Modes:
    on_peak = "on_peak"
    off_peak = "off_peak"
    mid_peak = "mid_peak"
    modes = [on_peak, mid_peak, off_peak]


class TimeOfUse(Resource):
    def __init__(self):
        Resource.__init__(self, "TimeOfUse", ["mode", "schedule"])

        try:
            config = Resource.resource("ConfigurationResource").config

            self.winter_schedule = config["time_of_use_schedule"]["winter"]
            self.summer_schedule = config["time_of_use_schedule"]["summer"]
            self.tou_off_peak_day = config["time_of_use_off_peak_day"]

            self.poller = resource_poll(self.update_schedule, MINS(1))
        except ResourceNotFoundException:
            print("TimeOfUse: No configuration resource!")

    def update_schedule(self):
        try:
            sched = None

            if is_schedule(self.winter_schedule, datetime.now().date()):
                sched = self.winter_schedule
                self.setValue("schedule", "winter")
            else:
                sched = self.summer_schedule
                self.setValue("schedule", "summer")

            if self.tou_off_peak_day != -1:
                if datetime.now().day == self.tou_off_peak_day:
                    self.mode = Modes.off_peak
                    self.set_value("mode", Modes.off_peak)
                    return

            for mode in Modes.modes:
                if is_mode(sched[mode], datetime.now().hour):
                    self.mode = mode
                    self.setValue("mode", mode)
                    break

            # print("time_of_use_resource: trying to update mode...")
            # print("time_of_use_resource: time of use mode is: {}".format(
            #    self.mode))

        except Exception:
            print("failed to determine power usage mode")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)


def main():
    winter_schedule = {
        "months": [11, 4],
        "on_peak": [[6, 10], [17, 20]],
        "mid_peak": [[10, 17], [20, 22]],
        "off_peak": [[22, 6]]
    }

    summer_schedule = {
        "months": [5, 10],
        "on_peak": [[15, 20]],
        "mid_peak": [[6, 15], [20, 22]],
        "off_peak": [[22, 6]]
    }

    assert is_schedule(winter_schedule, date(2018, 4, 1))
    assert is_schedule(winter_schedule, date(2018, 12, 1))
    assert not is_schedule(winter_schedule, date(2018, 5, 1))

    assert is_schedule(summer_schedule, date(2018, 5, 1))
    assert not is_schedule(summer_schedule, date(2018, 4, 1))

    assert is_mode(winter_schedule[Modes.on_peak], 6)
    assert not is_mode(winter_schedule[Modes.on_peak], 4)
    assert is_mode(winter_schedule[Modes.on_peak], 17)
    assert not is_mode(winter_schedule[Modes.on_peak], 16)
    assert is_mode(winter_schedule[Modes.off_peak], 4)
    assert is_mode(summer_schedule[Modes.off_peak], 22)
    assert is_mode(winter_schedule[Modes.off_peak], 22)
    assert not is_mode(summer_schedule[Modes.off_peak], 21)


if __name__ == "__main__":
    main()
