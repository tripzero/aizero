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

def is_schedule2(schedule, cur_date=None):

    if cur_date is None:
        cur_date = datetime.now()

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


def is_current_time_in_schedule(yaml_data, date_time=None):
    """
    Determines if the current time occurs during one of the schedules and time of use periods.

    Args:
        yaml_data (dict): Parsed YAML data containing schedules and TOU periods.

    Returns:
        bool: True if the current time matches any schedule and TOU period, False otherwise.
    """

    now = date_time 
    if now is None:
        now = datetime.now()

    current_month = now.month
    current_day = now.weekday()  # Monday is 0 and Sunday is 6
    current_hour = now.hour

    period_matches = []

    for schedule in yaml_data['schedules']:
        for schedule_name, schedule_data in schedule.items():
            # Check if the schedule is for specific months
            if 'months' in schedule_data and current_month not in schedule_data['months']:
                continue

            # Check if the schedule is for a specific day
            if 'day' in schedule_data and current_day != schedule_data['day']:
                continue

            # Check if the current time matches any TOU period
            for period_name, periods in schedule_data.items():
                if period_name in ['months', 'day']:
                    continue

                for period in periods:
                    start = period['start']
                    end = period['end']

                    if start <= current_hour < end or (start > end and (current_hour >= start or current_hour < end)):
                        period_matches.append(period_name)

    if len(period_matches):
        # We return the last matching schedule.
        return period_matches[-1]


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


class TimeOfUse2(Resource):
    def __init__(self, config):
        Resource.__init__(self, "TimeOfUse", ["mode", "schedule"])

        self.config = config
        try:
            self.schedules = config["schedules"]

            self.poller = resource_poll(self.update_schedule, MINS(1))
        except ResourceNotFoundException:
            print("TimeOfUse: No configuration resource!")

    def update_schedule(self):
        try:
            sched = None

            self.current_mode = is_current_time_in_schedule(self.config)
            self.set_value("mode", self.current_mode)

        except Exception:
            print("failed to determine time of use mode")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)


def test_time_of_use():
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


def test_time_of_use2():
    yaml_data = {
        "schedules": [
            {
                "reguler": {
                    "months": [1, 12],
                    "mid_peak": [{"start": 7, "end": 17}],
                    "off_peak": [{"start": 21, "end": 7}],
                    "on_peak": [{"start": 17, "end": 21}]
                }
            },
            {
                "sunday": {
                    "day": 6,
                    "off_peak": [{"start": 0, "end": 23}]
                }
            },
            {
                "saturday": {
                    "day": 5,
                    "off_peak": [{"start": 0, "end": 23}]
                }
            }
        ]
    }

    cur_mode = is_current_time_in_schedule(yaml_data)

    assert cur_mode is not None, "current TOU mode should not be None"

    test_mid_peak = datetime(2023, 12, 1, 8, 0)  # Regular mid_peak
    test_on_peak = datetime(2023, 12, 1, 18, 0) # Regular on_peak
    test_off_peak = datetime(2023, 12, 1, 22, 0) # Regular off_peak
    test_sat = datetime(2023, 12, 2, 10, 0) # Saturday off_peak
    test_sun = datetime(2023, 12, 3, 10, 0) # Sunday off_peak

    assert is_current_time_in_schedule(yaml_data, test_mid_peak) == "mid_peak"
    assert is_current_time_in_schedule(yaml_data, test_on_peak) == "on_peak"
    assert is_current_time_in_schedule(yaml_data, test_off_peak) == "off_peak"
    assert is_current_time_in_schedule(yaml_data, test_sat) == "off_peak"
    assert is_current_time_in_schedule(yaml_data, test_sun) == "off_peak"


if __name__ == "__main__":
    test_time_of_use2()
