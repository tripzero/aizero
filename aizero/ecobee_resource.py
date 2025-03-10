"""ecobee_resource.py

    resource for ecobee controller

"""

import os
import sys
import traceback
from datetime import datetime, timedelta

import shelve
import pytz

import asyncio
import numpy as np
import pyecobee as ecobee
from six.moves import input

from aizero.device_resource import DeviceResource, RuntimePriority
from aizero.occupancy_predictor import OccupancyPredictorResource
from aizero.resource import Resource, MINS, ResourceNameAlreadyTaken
from aizero.resource import ResourceNotFoundException
from aizero.resource_py3 import Py3Resource as resource_poll
from aizero.sys_time import get_current_datetime
from aizero.utils import run_thread


def f_to_c(f):
    return (f - 32) / 1.8


def c_to_f(c):
    return (c * 1.8) + 32


def _chk_resp(response, msg):
    if response.status.code != 0:
        raise Exception(msg)


class Programs:
    home = "home"
    away = "away"
    sleep = "sleep"
    hold = ""


class FanMode:
    on = "on"
    off = "off"
    auto = "auto"


class Ecobee:

    def __init__(self, thermostat_name, api_key):
        self.thermostat_name = thermostat_name
        self._global_occupancy = None

        home = os.environ["HOME"]

        self.thermostat = None

        self.db_file = "{}/.cache/{}_ecobee_db.db".format(
            home, self.thermostat_name)

        self.service = self.get_ecobee_service(thermostat_name, api_key)

        if not self.service.authorization_token:
            self.auth()

        if not self.service.access_token:
            self.request_tokens()

        now_utc = datetime.now(pytz.utc)
        if now_utc > self.service.refresh_token_expires_on:
            self.auth()
            self.request_tokens()
        elif now_utc > self.service.access_token_expires_on:
            token_response = self.refresh_tokens()

        self._running_program = None

        self._custom_program_map = {}

    def custom_program(self, program_name):
        if program_name in self._custom_program_map:
            return self._custom_program_map[program_name]

    def is_custom_program(self, cool_setpoint, heat_setpoint):
        """
            Returns custom program name if the cool_setpoint and
            heat_setpoint match or None
        """
        print("checking is_custom_program")
        print("cool point: {}".format(cool_setpoint))
        print("heat point: {}".format(heat_setpoint))

        for program, data in self._custom_program_map.items():
            print("program: {} cool: {}, heat: {}".format(
                program,
                data[0],
                data[1]))

            if (cool_setpoint == data[0] and
                    heat_setpoint == data[1]):
                return program

    def send_message(self, message):
        response = self.service.send_message(message)

        _chk_resp(thermostat_response,
                  'Failure while executing send_message:\n{0}'.format(
                      response.pretty_format()))

    @property
    def temperature(self):
        return self._temperature

    @property
    def humidity(self):
        return self._humidity

    @property
    def temperature_setpoint_cool(self):
        return self._setpoint_cool

    @property
    def temperature_setpoint_heat(self):
        return self._setpoint_heat

    @property
    def current_program(self):
        return self._running_program

    @property
    def global_occupancy(self):
        return self._global_occupancy

    @property
    def fan_mode(self):
        if self.thermostat is None:
            return None
        return self.thermostat.runtime.desired_fan_mode

    @property
    def cooling(self):
        # Assume that if temperature is higher than desired, hvac is running
        return self.temperature >= self.temperature_setpoint_cool

    def control_plug(self, plug_name, plug_state):
        if plug_state is True:
            plug_state = ecobee.PlugState.ON
        else:
            plug_state = ecobee.PlugState.OFF

        response = self.service.control_plug(plug_name, plug_state).status.code

        assert response == 0, 'Fail update_thermostats:\n{0}'.format(
            response.pretty_format())

    def set_hold(self, cool_temperature=None,
                 heat_temperature=None,
                 custom_program_name=None,
                 fan_mode=None):
        """
            Sets the hold temperature setpoint(s). Either cool_temperature or
            heat_temperature must be set. Default cool_temperature or
            heat_temperature is the current temperature setpoint if None.

            If custom_program_name is set the program name along with the
            setpoint settings will be saved.
        """

        if not cool_temperature:
            cool_temperature = self.temperature_setpoint_cool

        if not heat_temperature:
            heat_temperature = self.temperature_setpoint_heat

        if custom_program_name:
            self._custom_program_map[custom_program_name] = (
                cool_temperature,
                heat_temperature)

            print("custom program map: \n{}".format(self._custom_program_map))

        # temperature should be in Celcius. convert to F and multiply by 10
        cool_temperature = int(c_to_f(cool_temperature) * 10)
        heat_temperature = int(c_to_f(heat_temperature) * 10)

        reg_val = ecobee.SelectionType.REGISTERED.value

        params = {"holdType": "nextTransition",
                  "coolHoldTemp": cool_temperature,
                  "heatHoldTemp": heat_temperature}

        if fan_mode is not None:
            params["fan"] = fan_mode

        response = self.service.update_thermostats(
            ecobee.Selection(selection_type=reg_val,
                             selection_match=''),
            functions=[ecobee.Function("setHold", params)])

        if response.status.code != 0:
            raise Exception('Failure while executing set_hold:\n{0}'.format(
                response.pretty_format()))

    def set_program(self, program):
        # temporarily set the current program to this. update() will override
        # it with actual
        self._running_program = program

        if program in self._custom_program_map:
            cool_temp, heat_temp = self._custom_program_map[program]
            self.set_hold(cool_temp, heat_temp)
        else:
            response = self.service.set_hold(
                hold_climate_ref=program,
                hold_type=ecobee.HoldType.NEXT_TRANSITION)

            if response.status.code != 0:
                raise Exception('Failure set_program:\n{0}'.format(
                    response.pretty_format()))

    def resume_program(self, resume_all=True):
        print("resume_program called")
        # traceback.print_stack(file=sys.stdout)
        response = self.service.resume_program(resume_all=resume_all)

        if response.status.code != 0:
            raise Exception('Failure resume_program:\n{0}'.format(
                response.pretty_format()))

    def update(self):
        print("ecobee: update()")

        response = None

        try:
            reg_val = ecobee.SelectionType.REGISTERED.value
            response = self.service.request_thermostats(
                ecobee.Selection(selection_type=reg_val,
                                 selection_match='',
                                 include_device=True,
                                 include_runtime=True,
                                 include_sensors=True,
                                 include_program=True,
                                 include_extended_runtime=True,
                                 include_events=True,
                                 include_weather=False))

        except ecobee.exceptions.EcobeeApiException as e:
            print("error: EcobeeApiException")
            if e.status_code == 14:
                print("refreshing ecobee tokens...")
                self.refresh_tokens()
                return
            elif e.status_code == 16:
                self.auth()
                self.refresh_tokens()

        except Exception:
            print("error getting thermostats")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

            return

        print("ecobee: request_thermostats response received!")
        print(f"response code: {response.status.code}")

        if response.status.code != 0:
            print("ecobee: failure code... aborting update.")
            return

        if (response.thermostat_list is None or
                not len(response.thermostat_list)):
            raise Exception("No thermostats found")

        thermostat = None

        for dev in response.thermostat_list:
            if dev.name == self.thermostat_name:
                thermostat = dev
                break

        # we only support one thermostat...
        if not thermostat:
            print("ecobee: I can't find thermostate {}".format(
                self.thermostat_name))
            return

        self.thermostat = thermostat

        self._temperature = f_to_c(
            thermostat.runtime.actual_temperature / 10.0)
        self._humidity = thermostat.runtime.actual_humidity
        self._setpoint_cool = round(
            f_to_c(thermostat.runtime.desired_cool / 10.0), 1)
        self._setpoint_heat = round(
            f_to_c(thermostat.runtime.desired_heat / 10.0), 1)
        self._running_program = thermostat.program.current_climate_ref

        # running program may not be accurate if there is a running hold event
        # with a hold_climate_ref
        for event in thermostat.events:
            if event.running and event.type == 'hold':
                self._running_program = event.hold_climate_ref

        # finally, a hold might be a custom program, let's check
        custom_program = self.is_custom_program(self._setpoint_cool,
                                                self._setpoint_heat)

        if custom_program:
            self._running_program = custom_program

        self._global_occupancy = self.get_occupancy()

        return thermostat

    def get_sensors(self,
                    sensor_name=None,
                    capability=None,
                    capability_value=None,
                    debug=False):

        ret_sensors = []

        if self.thermostat is None:
            return ret_sensors

        sensors = self.thermostat.remote_sensors

        for sensor in sensors:
            if sensor_name and sensor.name != sensor_name:
                continue

            if capability:
                for sensor_capability in sensor.capability:
                    if debug:
                        print("searching: {}=={}".format(
                            sensor_capability.type,
                            capability))
                    if sensor_capability.type == capability:

                        if debug:
                            print("checking {}=={}".format(
                                capability_value,
                                sensor_capability.value))

                        if (capability_value and
                                sensor_capability.value != capability_value):
                            continue
                        ret_sensors.append(sensor)
            else:
                ret_sensors.append(sensor)

        return ret_sensors

    def get_sensor_value(self, sensor_name, capability, debug=False):
        sensors = self.thermostat.remote_sensors

        ret_sensors = []

        for sensor in sensors:
            if debug:
                print(f"searching for sensor {sensor_name} ?= {sensor.name}")

            if sensor_name and sensor.name != sensor_name:
                continue

            if debug:
                print("found!")

            for sensor_capability in sensor.capability:
                if debug:
                    print(f"searching for capability {sensor_capability} ?= {capability}")

                if sensor_capability.type == capability:
                    return sensor_capability.value

    def get_sensor_min_max(self, min_capability_filter=None,
                           min_capability_filter_value=None,
                           max_capability_filter=None,
                           max_capability_filter_value=None):

        sensors = self.get_sensors(
            capability=min_capability_filter,
            capability_value=min_capability_filter_value)

        if len(sensors) == 0:
            print("no sensors found matching criteria: {} == {}".format(
                min_capability_filter,
                min_capability_filter_value))
            return None, None

        max_sensors = self.get_sensors(
            capability=max_capability_filter,
            capability_value=max_capability_filter_value)

        if len(max_sensors) == 0:
            print("no sensors found matching criteria{} == {}".format(
                max_capability_filter,
                max_capability_filter_value))
            return None, None

        sensors.extend(max_sensors)

        sensor_temps = []
        sensors_with_temp = []

        for sensor in sensors:
            temperature = self.get_sensor_value(sensor.name,
                                                "temperature",
                                                debug=False)
            if temperature is not None:
                sensor_temps.append(temperature)
                sensors_with_temp.append(sensor)

        try:
            tmin = np.min(np.array(sensor_temps).astype(int))
            tmax = np.max(np.array(sensor_temps).astype(int))
        except ValueError as ve:
            print(ve)
            print(sensor_temps)
            return None, None

        print(f"get_sensor_min_max: {tmin}, {tmax}")

        return tmin, tmax

    def get_occupancy(self, sensor_name=None):
        sensors = self.thermostat.remote_sensors

        return len(self.get_sensors(sensor_name, "occupancy", "true")) > 0

    def get_ecobee_service(self, thermostat_name, api_key):

        ecobee_db = shelve.open(self.db_file, protocol=2)

        if thermostat_name in ecobee_db:
            ecobee_service = ecobee_db[thermostat_name]
            ecobee_db.close()
            return ecobee_service

        return ecobee.EcobeeService(thermostat_name=thermostat_name,
                                    application_key=api_key)

    def auth(self):
        response = self.service.authorize()

        print("ecobee pin => {}".format(response.ecobee_pin))

        input()

        self.persist_to_shelf()

    def request_tokens(self):
        response = self.service.request_tokens()

        self.persist_to_shelf()

    def refresh_tokens(self):
        try:
            print("ecobee: refreshing tokens")
            response = self.service.refresh_tokens()
            print(response)

            self.persist_to_shelf()
        except Exception as ex:
            print(f"Ecobee: refresh token exception: {ex}")

    def persist_to_shelf(self):
        pyecobee_db = shelve.open(self.db_file, protocol=2)
        pyecobee_db[self.service.thermostat_name] = self.service
        pyecobee_db.close()


class OccupancySensor(Resource):
    """
    Ecobee Occupancy Sensor
    this is really specific to ecobee.
    """

    def __init__(self, sensor_name, ecobee_service=None,
                 prediction_threshold=0.6):
        Resource.__init__(self, "OccupancySensor_{}".format(
            sensor_name), ["occupancy",
                           "predicted_occupancy",
                           "temperature"])

        self.sensor_name = sensor_name
        self.ecobee_service = ecobee_service

        if self.ecobee_service is None:
            # try to grab it from the EcobeeResource
            def wait_ecobee_service():
                self.ecobee_service = Resource.resource(
                    "EcobeeResource").ecobee_service

            Resource.waitResource("EcobeeResource", wait_ecobee_service)

            try:
                wait_ecobee_service()
            except ResourceNotFoundException:
                pass

        self.prediction_resource = OccupancyPredictorResource(
            self.name, prediction_threshold=prediction_threshold)

    async def poll(self):

        while True:
            if self.ecobee_service and self.ecobee_service.thermostat:
                value = self.ecobee_service.get_occupancy(self.sensor_name)
                # print("occupancy sensor {} is now: {}".format(
                #    self.name, value))
                self.set_value("occupancy", value)

                temp = self.ecobee_service.get_sensor_value(self.sensor_name,
                                                            "temperature")
                try:
                    if temp is not None:
                        self.set_value("temperature",
                                       f_to_c(float(temp) / 10.0))
                except ValueError as ve:
                    print("Failed to set temperature value")
                    print(ve)
                    print(temp)

            # prediction = self.prediction_resource.predict_occupancy()
            # print("occupancy sensor {} predicted is now: {}".format(
            #     self.name, prediction))
            # if prediction is not None:
            #     self.setValue("predicted_occupancy", round(prediction, 2))

            # try:
            #     import data_logging as dl
            #     dl.log_some_data("{}/occupancy_prediction_{}.json".format(
            #         self.prediction_resource.model_dir, self.name),
            #         predicted=round(prediction, 2),
            #         actual=self.getValue("occupancy"))
            # except Exception:
            #     pass

            await asyncio.sleep(MINS(3))


class EcobeeResource(DeviceResource):

    def __init__(self, occupancy_predictor_name=None, power_usage=1750):
        super().__init__("EcobeeResource",
                         power_usage=power_usage,
                         variables=["occupancy",
                                    "setpoint_heat",
                                    "setpoint_cool",
                                    "temperature",
                                    "humidity",
                                    "running_program",
                                    "temperature_delta",
                                    "occupied_temperature_delta"],
                         priority=RuntimePriority.high)

        self.subscribe("temperature", lambda v: self.process())
        self.subscribe("running_program", lambda v: self.process())

        self.occupancy_prediction = False
        self.occupancy_predictor_name = occupancy_predictor_name
        self.occupancy_predictor = None

        self.temperature_delta = 0
        self.occupied_temperature_delta = 0

        config = Resource.resource("ConfigurationResource").config

        api_key = config["ecobee_apikey"]
        thermostat_name = config["ecobee_thermostat_name"]

        self.ecobee_user_preferences = None
        self.present_users = []

        if "ecobee_user_preferences" in config:
            self.ecobee_user_preferences = config["ecobee_user_preferences"]

        self.ecobee_service = Ecobee(thermostat_name, api_key)

        self.poller = resource_poll(self.poll_func, MINS(3), is_coroutine=True)

        def wait_resources():
            try:
                self.occupancy_predictor = Resource.resource(
                    self.occupancy_predictor_name)
                self.occupancy_predictor.subscribe(
                    "occupancy_prediction", self.occupancy_changed)
            except ResourceNotFoundException:
                print(f"ecobee: WARNING: can't find occupancy predictor {self.occupancy_predictor_name}")

            Resource.resource("SolarPower").subscribe(
                "current_power", self.solar_power_changed)
            self.night_time_resource = Resource.resource("NightTime")

        required_resources = ["SolarPower", "NightTime"]

        if self.occupancy_predictor_name is not None:
            required_resources.append(self.occupancy_predictor_name)

        Resource.waitResource(required_resources, wait_resources)

        def wait_ble_resource():
            Resource.resource("BleUserResource").subscribe(
                "present_users", self.ble_present_users_changed)

        Resource.waitResource("BleUserResource", wait_ble_resource)

        self.ecobee_can_run_hold = False
        self.setpoint_cool = None
        self.setpoint_heat = None

    def ble_present_users_changed(self, value):
        self.present_users = value
        self.process()

    def occupancy_changed(self, value):
        self.occupancy_prediction = value
        self.process()

    def solar_power_changed(self, value):
        self.solar_power = value
        self.process()

    async def create_occupancy_resources(self):
        sensors = await run_thread(
            self.ecobee_service.get_sensors, capability="occupancy")

        for sensor in sensors:
            try:
                OccupancySensor(sensor.name)
            except ResourceNameAlreadyTaken:
                pass

    def process(self):
        asyncio.get_event_loop().create_task(self.do_process())

    async def do_process(self):
        print("ecobee processing: ")
        print("time: {}".format(get_current_datetime()))

        current_program = self.ecobee_service.current_program
        predict_time = (datetime.now() + timedelta(minutes=60))

        occupancy_prediction_60 = None

        op = self.occupancy_predictor

        if self.occupancy_predictor is not None:
            occupancy_prediction_60 = await run_thread(
                op.predict_occupancy,
                predict_time)

        # TODO: Why are we bailing out here?
        # if occupancy_prediction_60 is None:
        #     return

        is_night_time = self.night_time_resource.getValue("night_time")
        can_run = self.can_run()

        max_room_temp_delta = 4

        temp_min, temp_max = await run_thread(
            self.ecobee_service.get_sensor_min_max,
            max_capability_filter="occupancy",
            max_capability_filter_value="true")

        try:
            temp_min = f_to_c(float(temp_min) / 10)
            temp_max = f_to_c(float(temp_max) / 10)
            self.occupied_temperature_delta = abs(temp_min - temp_max)
        except TypeError as te:
            print(te)

        # also get the absolute temperature delta:
        abs_temp_min, abs_temp_max = await run_thread(
            self.ecobee_service.get_sensor_min_max)

        try:
            abs_temp_min = f_to_c(float(abs_temp_min) / 10)
            abs_temp_max = f_to_c(float(abs_temp_max) / 10)
            self.temperature_delta = abs(abs_temp_min - abs_temp_max)
        except TypeError as te:
            print(te)

        print("min: {} max: {}".format(temp_min, temp_max))

        print("max temp delta: {}".format(max_room_temp_delta))
        print(f"current delta: {self.temperature_delta}")
        print("current_program: {}".format(current_program))
        print("occupancy: {}".format(self.ecobee_service.global_occupancy))
        print("occupancy prediction: {}".format(self.occupancy_prediction))

        if occupancy_prediction_60 is not None:
            print("occupancy 60min prediction: {} at {}".format(
                round(occupancy_prediction_60, 2), predict_time))
        
        print("current running power: {}".format(
              round(self.device_manager.running_power, 2)))
        print("can run: {}".format(can_run))
        print("is running: {}".format(self.running()))
        print("fan mode: {}".format(self.ecobee_service.fan_mode))

        # get the typical home setpoints
        if current_program == Programs.home:
            self.setpoint_cool = self.ecobee_service.temperature_setpoint_cool
            self.setpoint_heat = self.ecobee_service.temperature_setpoint_heat

        """ if someone is home"""
        if self.ecobee_service.global_occupancy:
            if (not can_run and
                not is_night_time and
                current_program != Programs.hold and
                    current_program != "overbudget"):

                print("we really shouldn't be running hvac right now.")
                print("remaining capacity is only {}".format(
                    self.device_manager.remaining_power_capacity))
                print("our total usage is estimated: {}".format(
                    self.device_manager.running_power))

                await run_thread(
                    self.ecobee_service.set_hold,
                    self.ecobee_service.temperature_setpoint_cool + 0.5,
                    self.ecobee_service.temperature_setpoint_heat,
                    custom_program_name="overbudget")

            elif can_run and current_program == "overbudget":
                await run_thread(self.ecobee_service.resume_program,
                                 resume_all=False)

                print("ecobee: resuming program")

            elif current_program == Programs.away:
                await run_thread(self.ecobee_service.set_program,
                                 Programs.home)

                print("ecobee: set program 'home'")

            if (
                    temp_max is not None and
                    temp_min is not None and
                    abs(temp_max - temp_min) >= max_room_temp_delta):

                print("setting circulation program")
                await run_thread(self.ecobee_service.set_hold,
                                 fan_mode="on")

            elif (self.ecobee_service.fan_mode == "on"):
                await run_thread(self.ecobee_service.set_hold,
                                 fan_mode="auto")

        # if no one is home AND no one is usually home
        elif (not self.ecobee_service.global_occupancy and
              current_program != Programs.away and
              current_program != Programs.hold):

            # add self.occupancy_prediction is False when it works              

            await run_thread(self.ecobee_service.set_program,
                             Programs.away)
            print("ecobee: set program 'away'")

        # Is someone going to be home in the next 60 mins?
        elif (not self.ecobee_service.global_occupancy and
              occupancy_prediction_60 is not None and
              occupancy_prediction_60 >= 0.70 and
              current_program == Programs.away):

            await run_thread(self.ecobee_service.set_program,
                             Programs.home)
            print("ecobee: set program to 'Home'")

        """
        else:
            print("ecobee: resuming program")
            self.ecobee_service.resume_program()
        """

        current_program = self.ecobee_service.current_program

        # see if we need to modify the 'home' program according to
        # who is present
        # if ((self.setpoint_cool and self.setpoint_heat) and
        #         (current_program == Programs.home)):

        #     heat_mod, cool_mod = self.get_user_modifier()

        #     sp_cool = self.ecobee_service.temperature_setpoint_cool
        #     sp_heat = self.ecobee_service.temperature_setpoint_heat

        #     if heat_mod or cool_mod:
        #         if (self.setpoint_cool + cool_mod != sp_cool or
        #                 self.setpoint_heat + heat_mod != sp_heat):

        #             await run_thread(self.ecobee_service.set_hold,
        #                                   self.setpoint_cool + cool_mod,
        #                                   self.setpoint_heat + heat_mod)

    async def update_properties(self):
        print("Updating Ecobee properties...")
        self.set_value("occupancy", self.ecobee_service.global_occupancy)
        self.set_value("humidity", self.ecobee_service.humidity)
        self.set_value("temperature", self.ecobee_service.temperature)
        self.set_value("running_program",
                       self.ecobee_service.current_program)
        self.set_value("setpoint_cool",
                       self.ecobee_service.temperature_setpoint_cool)
        self.set_value("setpoint_heat",
                       self.ecobee_service.temperature_setpoint_heat)
        self.set_value("temperature_delta",
                       self.temperature_delta)
        self.set_value("occupied_temperature_delta",
                       self.occupied_temperature_delta)

    async def poll_func(self):
        try:
            await run_thread(self.ecobee_service.update)
            await self.create_occupancy_resources()

            await self.update_properties()

            # this is managed by hvac_cooler
            # TODO: we can reenable if service.cooling becomes more reliable
            # if self.ecobee_service.cooling:
            #    self.run()
            # else:
            #    self.stop()

        except Exception:
            print("failed to determine power usage mode")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=6, file=sys.stdout)

    def get_user_modifier(self):

        heat_mods = []
        cool_mods = []

        for user in self.present_users:
            if user not in self.ecobee_user_preferences:
                continue

            heat_mods.append(
                self.ecobee_user_preferences[user]["heat_setpoint_modifier"])
            cool_mods.append(
                self.ecobee_user_preferences[user]["cool_setpoint_modifier"])

        if not len(heat_mods):
            heat_mods = [0]

        if not len(cool_mods):
            cool_mods = [0]

        return max(heat_mods), min(cool_mods)
