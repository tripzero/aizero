import asyncio
import json
import logging
import numpy as np
import time

from aizero.resource import Resource, ResourceNotFoundException, MINS
from aizero.resource import get_resource as gr
from aizero.resource import ResourceRequires, HOURS
from aizero.remoteresource import RemoteRestResource
from aizero.resource_py3 import Py3PollWhileTrue as poll_while_true
from aizero.resource_py3 import Py3Resource as resource_poll


class RuntimePriority:
    """ RuntimePriority is used to automatically run or stop devices to
        maintain power budget.  When a higher priority devices requests to run,
        Device manager may stop a lower priority Device to keep within quota.
        A priority of None will not be automatically managed by device manager.

        Devices that set priority of med and low must have the ability to stop
        when stop() is called.
    """
    high = 1
    medium = 5
    low = 10


class RuntimePolicies:
    """ Policy: none
        No policy is defined. Any conditions will still be checked before
        device can run.
    """
    none = "none"

    """ Policy: GroupPolicy
        policy containing several other policies
    """
    group = "policy_group"

    """ Policy: run_if_can
        Device will run if can_run() returns True.  This is identical to
        conditions = [device.can_run].
    """
    run_if_can = "run_if_can"

    """ Policy: dim_if_cannot
        Device will dim to dim_level if can_run() returns False
    """
    dim_if_cannot = "dim_if_cannot"

    """ Policy: off_if_unoccupied
        stop() called on device where occupancy is False
    """
    off_if_unoccupied = "off_if_unoccupied"

    """ Policy: link_device_policy
        can_run() is True when linked device is running
    """
    link_device_policy = "link_device_policy"

    """ Policy: run_if_temperature
        can_run() is True if temperature is above specified value
    """
    run_if_temperature = "run_if_temperature"

    @classmethod
    def policy(self, policy_name, **kwargs):
        """
        returns policy instance for matching policy_name or None
        """
        # TODO: implement
        pass


class RuntimePolicy:
    """
        accepts a policy and a set of conditions.  conditions are a list of
        callables that return either True or False.  If all the conditions are
        True, run() will be called on the device.  A conditionthat returns
        False will trigger the stop() call on the device.

        A condition can return None, which indicates no action.
    """

    def __init__(self, policy=RuntimePolicies.none,
                 conditions=None, or_conditions=None):
        if conditions is None:
            conditions = []

        if or_conditions is None:
            or_conditions = []

        self.policy = policy
        self.conditions = conditions
        self.or_conditions = or_conditions

    def process(self, device):
        pass

    def has_conditions(self):
        """ return if there are conditions or not. if this returns False,
        run_conditions() is not called.
        """
        return len(self.conditions) > 0 or len(self.or_conditions) > 0

    def run_conditions(self):

        ret_val = True
        do_something = False

        if not len(self.conditions):
            ret_val = False

        for condition in self.conditions:
            cond_ret_val = condition()

            # print("running condition from {}: {}: {}".format(
            #    self.policy, condition.__name__, cond_ret_val))

            do_something |= cond_ret_val is not None

            if cond_ret_val is not None:
                ret_val &= cond_ret_val

        or_ret_val = False

        for condition in self.or_conditions:
            cond_ret_val = condition()

            # print("running 'or' condition from {}: {}: {}".format(
            #    self.policy, condition.__name__, cond_ret_val))

            do_something |= cond_ret_val is not None

            if cond_ret_val is not None:
                or_ret_val |= cond_ret_val

        if do_something:
            return ret_val or or_ret_val

    def to_json(self):
        serialized = {}
        serialized["name"] = self.policy
        conditions = []

        for cond in self.conditions:
            conditions.append(cond.__name__)

        serialized["conditions"] = conditions

        return json.dumps(serialized)


class PolicyGroup(RuntimePolicy):

    def __init__(self, policies=None, or_policies=None):

        if policies is None:
            policies = []

        if or_policies is None:
            or_policies = []

        self.policies = policies
        self.or_policies = or_policies

        conditions = []

        for policy in policies:
            if not isinstance(policy, RuntimePolicy):
                raise ValueError("RuntimePolicy got {} in GroupPolicy".format(
                    policy))

            conditions.append(policy.run_conditions)

        or_conditions = []

        for policy in or_policies:
            if not isinstance(policy, RuntimePolicy):
                raise ValueError("RuntimePolicy got {} in GroupPolicy".format(
                    policy))

            or_conditions.append(policy.run_conditions)

        super().__init__(policy=RuntimePolicies.group,
                         conditions=conditions,
                         or_conditions=or_conditions)

    def process(self, device):
        for policy in self.policies:
            policy.process(device)

        for policy in self.or_policies:
            policy.process(device)

    def to_json(self):
        serialized = json.loads(super().to_json())

        serialized["policies"] = [policy.to_json()
                                  for policy in self.policies]
        serialized["or_policies"] = [policy.to_json()
                                     for policy in self.or_policies]

        return json.dumps(serialized)


class RunIfCanPolicy(RuntimePolicy):

    def __init__(self, conditions=None, or_conditions=None):
        if conditions is None:
            conditions = []

        super().__init__(policy=RuntimePolicies.run_if_can,
                         conditions=conditions,
                         or_conditions=or_conditions)

        self.conditions.append(self.can_run)
        self.device = None

    def process(self, device):
        if not self.device:
            self.device = device

    def can_run(self):

        return self.device.can_run()


class DimIfCannotPolicy(RuntimePolicy):
    """
        Dim the light to dim_level if can_run() is False
    """

    def __init__(self, dim_level=50, conditions=None, or_conditions=None):
        if conditions is None:
            conditions = []

        super().__init__(policy=RuntimePolicies.dim_if_cannot,
                         conditions=conditions,
                         or_conditions=or_conditions)

        self.dim_level = dim_level

    def process(self, device):
        if not device.can_run():
            device.dimmer_level = self.dim_level
        else:
            device.dimmer_level = 100


class OffIfUnoccupied(RuntimePolicy):
    """
        calls set(False) on device if the assigned OccupancyResource has
        occupancy of False.
        Requires "occupancy" or "predicted_occupancy" property be available
        on resource.
    """

    def __init__(self, occupancy_resource_name, conditions=None,
                 or_conditions=None,
                 occupancy_property="occupancy"):

        if conditions is None:
            conditions = []

        super().__init__(policy=RuntimePolicies.off_if_unoccupied,
                         conditions=conditions,
                         or_conditions=or_conditions)

        self.checked = False
        self.occupancy = None

        def wait_occupancy(gr):
            self.occupancy_resource = gr(occupancy_resource_name)
            self.conditions.append(self.can_run)

            if self.occupancy_resource.hasProperty(occupancy_property):
                self.occupancy = self.occupancy_resource.subscribe2(
                    occupancy_property)
            else:
                raise ValueError(
                    """resource {} does not {} property""".format(
                        occupancy_resource_name, occupancy_property))

        self.rsrcs = ResourceRequires([occupancy_resource_name],
                                      wait_occupancy)

    @asyncio.coroutine
    def can_run_check_timeout(self):
        self.checked = True

        yield from asyncio.sleep(MINS(15))

        self.checked = False

    def can_run(self):
        if self.checked:
            return

        is_occupied = self.occupancy.value
        logging.debug("{} is occupied: {}".format(self.occupancy.resource.name,
                                                  is_occupied))

        logging.debug(self.occupancy.variable)

        if is_occupied is not None and not is_occupied:
            asyncio.get_event_loop().create_task(self.can_run_check_timeout())
            logging.debug("condition returning False")
            return False

    def to_json(self):
        """
        overrides RuntimePolicy.to_json and adds name of occupancy resource
        associated with this policy
        """

        serialized = json.loads(super().to_json())

        serialized["associated_sensor"] = json.loads(
            self.occupancy_resource.to_json())

        return json.dumps(serialized)


class OnIfOccupied(OffIfUnoccupied):
    """
        calls set(False) on device if the assigned OccupancyResource has
        occupancy of False.
        Requires "occupancy" or "predicted_occupancy" property be available
        on resource.
    """

    def __init__(self, occupancy_resource_name, conditions=None,
                 or_conditions=None,
                 occupancy_property="occupancy"):

        if conditions is None:
            conditions = []

        super().__init__(occupancy_resource_name,
                         conditions=conditions,
                         or_conditions=or_conditions,
                         occupancy_property=occupancy_property)

    def can_run(self):
        if self.checked:
            return

        is_occupied = self.occupancy.value
        logging.debug("{} is occupied: {}".format(self.occupancy.resource.name,
                                                  is_occupied))

        logging.debug(self.occupancy.variable)

        if is_occupied is not None and is_occupied:
            asyncio.get_event_loop().create_task(self.can_run_check_timeout())
            logging.debug("condition returning False")
            return True


class LinkDevicePolicy(RuntimePolicy):
    """
    Links a device with another device.  When the other device is running,
    this will too.
    """

    def __init__(self, linked_device_name, conditions=None,
                 or_conditions=None):
        super().__init__(policy=RuntimePolicies.link_device_policy,
                         conditions=conditions,
                         or_conditions=or_conditions)

        self.linked_device_running = None

        def wait_resource():
            self.linked_device_running = gr(
                linked_device_name).subscribe2("running")

        try:
            gr(linked_device_name)
            wait_resource()
        except ResourceNotFoundException:
            Resource.waitResource(linked_device_name, wait_resource)

    def _pass_process(self, device):
        pass

    def process(self, device):
        if (self.can_run not in self.conditions and
            self.linked_device_running is not None and
                self.linked_device_running.value is not None):

            self.conditions.append(self.can_run)

            # we are ready. no more need for process to do anything
            self.process = self._pass_process

    def can_run(self):
        return self.linked_device_running.value

    def to_json(self):
        """
        overrides RuntimePolicy.to_json and adds name of resource
        associated with this policy
        """

        linked_name = self.linked_device_running.resource.name

        serialized = json.loads(super().to_json())
        serialized["linked_resource"] = linked_name

        return json.dumps(serialized)


class RunIfTemperaturePolicy(RuntimePolicy):

    def __init__(self, sensor_name, set_point, conditions=None,
                 or_conditions=None):
        if conditions is None:
            conditions = []

        super().__init__(policy=RuntimePolicies.run_if_temperature,
                         conditions=conditions,
                         or_conditions=or_conditions)

        self.set_point = set_point
        self.temperature = None
        self.sensor_name = sensor_name

        def wait_resource(gr):
            self.temperature = gr(sensor_name).subscribe2("temperature")

            self.conditions.append(self.can_run)

        self.rsrcs = ResourceRequires([sensor_name],
                                      wait_resource)

    def can_run(self):

        if self.temperature is not None:
            # print("sensor_name={}".format(self.sensor_name))
            # print("temperature={}".format(self.temperature.value))
            return self.temperature.value > self.set_point
        else:
            print("temperature is None. This is probably a bug")

    def to_json(self):
        """
        overrides RuntimePolicy.to_json and adds name of resource
        associated with this policy
        """

        serialized = json.loads(super().to_json())
        serialized["associated_sensor"] = self.sensor_name
        serialized["setpoint"] = self.set_point

        return json.dumps(serialized)


class MinimumRuntime(RuntimePolicy):
    """ Creates a condition that ruturns true if running time of device
        is less than the minimum specified running time.
    """

    def __init__(self, device, min_runtime, **kwargs):
        """
        args:
            device: device to monitor running time
            min_runtime: minimum running time in seconds.
        """
        super().__init__(policy="minimum_runtime", **kwargs)

        self.device = device
        self.min_runtime = min_runtime

        self.current_runtime = 0
        self.last_time = None

        self.conditions.append(self.can_run)

        asyncio.get_event_loop().create_task(self.reset())

    @asyncio.coroutine
    def reset(self):
        while True:
            yield from asyncio.sleep(HOURS(24))

            self.current_runtime = 0

    def process(self, device):
        running = device.running()

        if running and self.last_time is None:
            print("starting current runtime")
            self.last_time = time.monotonic()
            return

        if running:
            self.current_runtime += time.monotonic() - self.last_time
            self.last_time = time.monotonic()

        elif not running and self.last_time is not None:
            print("stopping current runtime!")
            self.current_runtime += time.monotonic() - self.last_time
            self.last_time = None

    def can_run(self):
        if self.current_runtime < self.min_runtime:
            return True

    def to_json(self):
        """
        overrides RuntimePolicy.to_json and adds name of resource
        associated with this policy
        """

        serialized = json.loads(super().to_json())
        serialized["current_runtime"] = self.current_runtime
        serialized["min_runtime"] = self.min_runtime

        return json.dumps(serialized)


class TimeOfUsePolicy(RuntimePolicy):

    def __init__(self, time_of_use_mode, **kwargs):
        super().__init__(policy="time_of_use", **kwargs)

        self.time_of_use_mode = time_of_use_mode

        self.rsrcs = ResourceRequires(["TimeOfUse"],
                                      self.start)

    def start(self, rsrcs):
        self.conditions.append(self.can_run)

    def can_run(self):
        tou = self.rsrcs("TimeOfUse")
        return self.time_of_use_mode == tou.get_value("mode")


class DeviceManager(Resource):

    def __init__(self, name="DeviceManager", max_power_budget=0,
                 power_source=None, debug=False):
        """ max_power_budget is to be used to define the max power budget when
            there is no solar system (ie battery system, or pure-grid system).

            power_source must have the "current_power" property unless max
            power_budget is set.

            available_power is max_power_budget until power_source updates.
        """

        super().__init__(name, ["total_power", "running_devices",
                                "capacity", "total_capacity"])

        if power_source is None:
            power_source = ["SolarPower"]

        power_sources = power_source

        self.running_devices = []
        self.managed_devices = []

        self.max_power_budget = max_power_budget
        self.time_of_use_mode = None
        self.debug = debug

        self.power_sources = None

        if not max_power_budget:
            self.power_sources = ResourceRequires(
                power_sources, lambda rsrcs: True)

        def wait_time_of_use():
            gr("TimeOfUse").subscribe("mode", self.time_of_use_changed)

        try:
            wait_time_of_use()
        except ResourceNotFoundException:
            Resource.waitResource("TimeOfUse", wait_time_of_use)

        self.poller = resource_poll(self.process_managed_devices, MINS(1))

    def time_of_use_changed(self, value):
        self.time_of_use_mode = value
        # self.process_managed_devices()

    def register_managed_device(self, device):
        self.debug_print("registering managed device: {}".format(device.name))
        if device not in self.managed_devices:
            self.managed_devices.append(device)
            device.subscribe("power_usage", self.device_power_usage_changed)

    @property
    def highest_consuming_managed_device(self):
        pigs = {}

        for device in self.managed_devices:
            pigs[device.power_usage] = device

        return pigs[np.max(list(pigs.keys()))]

    @property
    def available_power(self):

        if self.power_sources is None:
            return self.max_power_budget

        ap = 0

        for power_source in self.power_sources.resources():
            cp = power_source.get_value("current_power")

            if cp is None:
                continue

            ap += cp

        return ap

    @property
    def capacity(self):
        return self.available_power

    @property
    def utilization(self):
        return round(self.running_power / max(self.available_power, 1) * 100.0, 1)

    @property
    def remaining_power_capacity(self):
        return self.available_power - self.running_power

    @property
    def capcity_percentage(self):
        return min(round(self.running_power /
                         max(self.available_power, 1) * 100.0, 1), 100)

    def debug_print_capacity(self):
        cap_per = self.capcity_percentage

        print("total budget: {}W".format(self.available_power))
        print("total usage: {}W".format(round(self.running_power), 1))
        print("capacity: {}%".format(
            cap_per))

        self.set_value("capacity", cap_per)

    def debug_print(self, msg):
        if self.debug:
            print(msg)

    def process_managed_devices(self):
        cap_per = self.capcity_percentage

        self.set_value("capacity", cap_per)
        self.set_value("total_capacity", self.capacity)

        self.debug_print("managed devices: {}".format(
            self.managed_devices_pretty))

        for device in self.managed_devices:

            self.debug_print(
                "processing managed device: {}".format(device.name))

            has_any_conditions = False

            # set to None at first because there may be no action
            can_run = None

            can_run_policies = []
            can_not_run_policies = []

            for policy in device.runtime_policy:
                if policy is None:
                    print("{} has a NoneType policy".format(device.name))
                    continue

                policy.process(device)

                has_any_conditions |= policy.has_conditions()

                if policy.has_conditions():
                    cond_vals = policy.run_conditions()

                    self.debug_print("cond_vals for {}: {}".format(
                        cond_vals,
                        policy.policy))

                    if cond_vals is True:
                        can_run_policies.append(policy.policy)

                    elif cond_vals is False:
                        can_not_run_policies.append(policy.policy)

                    # run_conditions() may not specify an action by returning
                    # None:
                    if cond_vals is not None:
                        if can_run is None:
                            can_run = True

                        can_run &= cond_vals

                    else:
                        self.debug_print("cond_vals: {}".format(cond_vals))

                    self.debug_print("can_run? {}".format(can_run))

                else:
                    self.debug_print(
                        "policy {} and no conditions".format(policy.policy))

            self.debug_print("has conditions: {}".format(has_any_conditions))
            self.debug_print("can_run: {}".format(can_run))

            if has_any_conditions and can_run:
                if not device.running():
                    self.debug_print_capacity()
                    print("device {} can run. {} W. calling run()".format(
                        device.name, device.max_power_usage))
                    print("estimated power usage will be: {}W".format(
                        self.running_power + device.max_power_usage))
                    print("policies that say this device can run:")
                    print("\n".join(can_run_policies))
                    device.run()

            elif has_any_conditions and can_run is False:
                if device.running():
                    print("device {} can NOT run. {} W. calling stop()".format(
                        device.name, device.power_usage))
                    self.debug_print_capacity()
                    print("policies stopping this device:")
                    print("\n".join(can_not_run_policies))
                    device.stop()
                    # self.debug_print_capacity()
            elif has_any_conditions and can_run is None:
                self.debug_print("can_run is None")
            elif not has_any_conditions:
                self.debug_print("has_any_conditions is False")
            else:
                raise Exception("Unknown state")

    def device_power_usage_changed(self, value):
        pass
        # self.process_managed_devices()

    @property
    def running_power(self):
        total_power = 0

        for dev in self.running_devices:
            total_power += dev.power_usage

        return total_power

    def is_running(self, device):
        return device in self.running_devices

    def is_higher_priority(self, device, exclude_devices_list=[]):
        """ return first device that has a "lower" priority or None
        """

        if not device.priority:
            return

        for dev in self.running_devices:
            if (dev.priority and device.priority < dev.priority and
                    dev not in exclude_devices_list):
                return dev

    def kickable_devices(self, device, requested_power, strict=False):
        """ strict means return None if no amount of kicked devices will meet
            budget criteria
        """

        def get_power_savings(devices):
            all_power_usage = 0

            for d in devices:
                all_power_usage += d.power_usage

            return all_power_usage

        _devices_we_can_kick = []
        dev = self.is_higher_priority(
            device, exclude_devices_list=_devices_we_can_kick)

        while dev is not None:
            _devices_we_can_kick.append(dev)

            if get_power_savings(_devices_we_can_kick) >= requested_power:
                return _devices_we_can_kick

            dev = self.is_higher_priority(
                device, exclude_devices_list=_devices_we_can_kick)

        if len(_devices_we_can_kick) and not strict:
            return _devices_we_can_kick

    def overrun_amount(self, device_to_run=None):
        power_delta = self.running_power - self.available_power

        # if this device is already "running", it's power is already included
        if device_to_run and not self.is_running(device_to_run):
            power_delta = (self.running_power +
                           device_to_run.max_power_usage) - self.available_power

        if device_to_run:
            return max(0, min(power_delta, device_to_run.max_power_usage))

        return max(0, power_delta)

    def can_run(self, device):
        self.debug_print("can_run: device: {}".format(device.name))

        over_budget_amount = self.overrun_amount(device)
        cr = over_budget_amount == 0 and self.capcity_percentage < 100

        self.debug_print("can_run: device under budget: {}".format(cr))

        self.debug_print(
            "can_run: over budget amount: {}".format(over_budget_amount))

        self.debug_print("can_run: capacity: {}".format(
            self.capcity_percentage))

        is_time_of_use_mode = False
        if self.time_of_use_mode is not None:
            for mode in device.runtime_modes:
                is_time_of_use_mode |= mode == self.time_of_use_mode

        cr |= is_time_of_use_mode

        self.debug_print(
            "can_run: is time of use mode: {}".format(is_time_of_use_mode))

        has_kickable = self.kickable_devices(device, over_budget_amount, True)

        # finally, check if there are lower priority devices we can bump:
        cr |= has_kickable is not None

        self.debug_print("can_run: has kickable? {}".format(has_kickable))

        return cr

    def started_running(self, device):
        # first try to make room for device by stopping lower priority devices
        # until we meet our power budget:
        overrun = self.overrun_amount(device)
        if overrun:
            devices_to_kick = self.kickable_devices(device, overrun)

            if devices_to_kick is not None:
                for device_to_kick in devices_to_kick:
                    print("device_manager: kicking device: {}".format(
                        device_to_kick.name))
                    device_to_kick.stop()

        if device not in self.running_devices:
            print("device: {} started running".format(device.name))
            # traceback.print_stack(file = sys.stdout)
            self.running_devices.append(device)

        self.setValue("running_devices", self.running_devices_pretty)
        self.setValue("total_power", self.running_power)

    def finished_running(self, device):
        if device in self.running_devices:
            self.running_devices.remove(device)

        self.setValue("running_devices", self.running_devices_pretty)
        self.setValue("total_power", self.running_power)

        # process devicess because total power changed
        self.process_managed_devices()

    @property
    def running_devices_pretty(self):
        rdp = []
        for device in self.running_devices:
            rdp.append(device.name)

        return rdp

    @property
    def managed_devices_pretty(self):
        rdp = []
        for device in self.managed_devices:
            rdp.append(device.name)

        return rdp


class DeviceResource(Resource):

    def __init__(self, name, power_usage=0, runtime_modes=None, variables=None,
                 priority=None, runtime_policy=None,
                 device_manager="DeviceManager", resource_args=None):
        """ :param power_usage how much power this device consumes in Watts
            :param runtime_modes time-of-use modes this device can run in. if
             runtime_modes == [], device can run in all runtime modes

            :param runtime_policy policies that determines whether this device
             will run or not. if runtime_policy == None, DeviceManager will
             not manage the device
            :param priority the priority of the device. if not set,
             DeviceManager will not manage the device (see RuntimePriority)
        """
        if runtime_modes is None:
            runtime_modes = []

        if variables is None:
            variables = []

        if runtime_policy is None:
            runtime_policy = []

        if resource_args is None:
            resource_args = {}

        vars = ["power_usage", "runtime_modes", "priority", "running"]

        for variable in variables:
            if variable not in vars:
                vars.append(variable)

        super().__init__(name, vars, **resource_args)

        self.device_manager = None

        self.update_power_usage(power_usage)
        self._max_power_usage = power_usage

        self.setValue("power_usage", self._power_usage)

        self.runtime_modes = runtime_modes
        self.setValue("runtime_modes", self.runtime_modes)

        self.priority = priority
        self.setValue("priority", priority)

        self.run_request_count = 0
        self.run_requests = {}

        if not isinstance(runtime_policy, list):
            self._runtime_policy = [runtime_policy]
        elif runtime_policy is None:
            self._runtime_policy = []
        else:
            self._runtime_policy = runtime_policy

        def register_with_manager(rsrcs):
            self.device_manager = rsrcs(device_manager)
            if self.runtime_policy:
                self.device_manager.register_managed_device(self)

        self.rsrcs = ResourceRequires([device_manager], register_with_manager)

    @property
    def runtime_policy(self):
        return self._runtime_policy

    @runtime_policy.setter
    def runtime_policy(self, value):
        self.set_runtime_policy(value)

    def set_runtime_policy(self, value):
        print("can has runtime_policy")
        if not isinstance(value, list):
            self._runtime_policy = [value]
        else:
            self._runtime_policy = value

        if self.device_manager and value:
            self.device_manager.register_managed_device(self)

    def running(self):
        return self.device_manager and self.device_manager.is_running(self)

    @property
    def power_usage(self):
        return self._power_usage

    @power_usage.setter
    def power_usage(self, value):
        self.update_power_usage(value)

    def update_power_usage(self, value):
        self._power_usage = value
        self.setValue("power_usage", value)

    @property
    def max_power_usage(self):
        self._max_power_usage = max(self._power_usage, self._max_power_usage)

        return self._max_power_usage

    def wait_can_run(self, callback=None, interval=1):
        def wait_func():

            if not self.can_run():
                return True

            self.run()

            if callback:
                callback()

            return False

        poll_while_true(wait_func, interval)

    def can_run(self):
        return self.device_manager and self.device_manager.can_run(self)

    def request_run(self):
        """
        Request that this device run. Returns request id to be used to remove
        the request.  Device will continue running until all requests have been
        removed via the remove_request_run() method.

        :see remove_request_run
        :returns request id
        """
        self.run_request_count += 1

        self.run_requests[self.run_request_count] = True

        if not self.running():
            self.run()

        return self.run_request_count

    def remove_request_run(self, id):
        """
        remove a previous run request made with the request_run() method.
        remove_request_run() will call stop()

        param: id identification handle returned from request_run
        """

        if id not in self.run_requests:
            return

        del self.run_requests[id]

        if self.running():
            self.stop()

    def run(self):
        """
        Register with the device manager as running
        """
        if self.device_manager is None:
            return False

        self.device_manager.started_running(self)
        self.setValue("running", True)

        return True

    def stop(self):
        """
        Remove device from the device manager's running list.

        This should be called first in any subclass stop() implementation.
        The device should only stop running if this returns True.

        :return False if cannot or should not be stopped
        :return True if successfully removed from device manager's running list
        """

        if not self.can_stop():
            return False

        self.device_manager.finished_running(self)
        self.setValue("running", False)

        return True

    def can_stop(self):
        """
        Return True if the device can stop.

        The device may not be able to stop if there are run requests that have
        not been removed.
        """

        if len(self.run_requests):
            return False

        if not self.device_manager:
            return False

        return True

    def to_json(self):
        """
        overrides Resource.to_json and adds policy data
        """

        serialized = json.loads(super().to_json())

        policies_serialized = []

        for policy in self._runtime_policy:
            policies_serialized.append(json.loads(policy.to_json()))

        serialized["policies"] = policies_serialized

        return json.dumps(serialized)


class RemoteRestDeviceResource(RemoteRestResource, DeviceResource):

    """
        variable_map = {"local_variable_name" : "remote_variable_name"}
    """

    def __init__(self, name, hammock_instance, variable_map=None, **kwargs):

        if variable_map is None:
            variable_map = {}

        RemoteRestResource.__init__(self, name, hammock_instance)

        self.variable_map = variable_map
        self.variables = list(self.variables.keys())
        self.variables.extend(variable_map.keys())

        if 'variables' in kwargs:
            self.variables.extend(kwargs["variables"])

        kwargs["variables"] = self.variables

        DeviceResource.__init__(self, name, **kwargs)

    def poll_func(self):
        if "power_usage" not in self.variables:
            return  # not ready yet

        RemoteRestResource.poll_func(self)

        for variable in self.variable_map:
            self.setValue(variable, self.getValue(self.variable_map[variable]))

        self.power_usage = self.getValue("power_usage")

        if self.power_usage:
            DeviceResource.run(self)
        else:
            DeviceResource.stop(self)


def test_main():
    Resource.clearResources()
    # test priority system
    device_manager = DeviceManager(max_power_budget=800)

    device_low = DeviceResource(
        "LowPriority", 200, priority=RuntimePriority.low)
    device_none = DeviceResource("NonePriority", 200)
    device_med = DeviceResource(
        "MediumPriority", 200, priority=RuntimePriority.medium)

    device_high = DeviceResource(
        "HighPriority", 200, priority=RuntimePriority.high)

    device_high_too_much_power = DeviceResource(
        "HighPriority2", 400, priority=RuntimePriority.high)

    device_low_too_much_power = DeviceResource(
        "LowPriority2", 400, priority=RuntimePriority.low)

    device_low.run()
    device_med.run()
    device_none.run()

    assert device_low.running
    assert device_med.running
    assert device_none.running

    assert device_manager.running_power == 600
    assert device_high.can_run()

    power_delta = device_manager.overrun_amount(device_low_too_much_power)

    assert power_delta, "power delta {} should be > 0".format(power_delta)

    assert not device_low_too_much_power.can_run()

    power_delta = device_manager.overrun_amount(device_high_too_much_power)
    print(power_delta)

    assert power_delta
    assert device_manager.is_higher_priority(device_med)
    assert device_manager.is_higher_priority(device_high)
    assert device_manager.is_higher_priority(device_high_too_much_power)

    # There should be a couple devices we can kick
    kickables = device_manager.kickable_devices(
        device_high_too_much_power, power_delta)

    print("kickables: {}".format(kickables))
    assert kickables is not None

    assert device_none not in device_manager.kickable_devices(
        device_high_too_much_power, power_delta)
    assert device_low in device_manager.kickable_devices(
        device_high_too_much_power, power_delta)

    device_high_too_much_power.run()

    assert not device_low.running()
    assert device_high_too_much_power.running()


"""def test_remote():
    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=800)

    from hammock import Hammock

    remote_resource = RemoteRestDeviceResource(
        "GreenhouseDeviceManager",
        Hammock(""),
        variable_map={
            "power_usage": "total_power"},
        device_manager=device_manager)

    assert "total_power" in remote_resource.variables

    # run the even loop so "remote_resource.poll" is hit
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    assert remote_resource.getValue(
        "total_power") == remote_resource.getValue("power_usage")
"""


def test_occupancy_policy():
    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=800)

    occupancy_resource = Resource(
        "SomeOccupancyThing", variables=["occupancy"])

    occupancy_resource.setValue("occupancy", True)

    off_policy = OffIfUnoccupied(occupancy_resource.name)

    dev1 = DeviceResource("dev1", power_usage=100,
                          device_manager=device_manager.name,
                          runtime_policy=[off_policy])

    dev1.run()

    device_manager.process_managed_devices()

    assert dev1.running(), "Device should be running"

    occupancy_resource.setValue("occupancy", False)

    assert off_policy.occupancy.value is False

    device_manager.process_managed_devices()

    assert not dev1.running(), "Device should not be running"

    occupancy_resource.setValue("occupancy", True)

    device_manager.process_managed_devices()

    assert not dev1.running(), "Device should not be running"


def test_runifcan():
    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=1000)

    device1 = DeviceResource("my special device", power_usage=500)
    device2 = DeviceResource("my other device", power_usage=500)

    device1.run()
    device2.run()

    assert device1.running()
    assert device2.running()

    device4 = DeviceResource("automated device",
                             power_usage=100,
                             runtime_policy=RunIfCanPolicy())

    assert not device4.running()

    device1.stop()

    assert device4.running()


def test_zero_power_usage_can_run():

    Resource.clearResources()

    DeviceManager(max_power_budget=800)
    device_1 = DeviceResource("Power Hog 3000", 800)

    device_1.run()

    device_2 = DeviceResource("Power Hog 3001", 0)

    assert not device_2.can_run()


def test_group_policy():

    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=800)

    policy1 = RuntimePolicy(conditions=[lambda: True])
    policy2 = RuntimePolicy(conditions=[lambda: False])

    and_group = PolicyGroup(policies=[policy1, policy2])
    or_group = PolicyGroup(policies=[policy1], or_policies=[policy2])

    fake_device = DeviceResource("fake-device", 800)
    fake_device_or = DeviceResource("fake-device-or", 800)

    fake_device.set_runtime_policy([and_group])
    fake_device_or.set_runtime_policy([or_group])

    fake_device.run()
    fake_device_or.run()

    device_manager.process_managed_devices()

    assert not fake_device.running()
    assert fake_device_or.running()


def test_minimum_runtime():
    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=800, debug=True)

    device = DeviceResource("fake_device", 2000)

    policy1 = MinimumRuntime(device, 1)
    policy2 = RunIfCanPolicy()

    group = PolicyGroup(or_policies=[policy1, policy2])

    device.set_runtime_policy([group])

    device.run()

    device_manager.process_managed_devices()

    assert device.running()
    assert policy1.last_time is not None

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

    device_manager.process_managed_devices()

    print(device.to_json())

    assert policy1.current_runtime > 1
    assert not policy1.can_run()
    assert not policy2.can_run()
    assert not policy1.run_conditions()
    assert not policy2.run_conditions()
    assert not device.running()


def test_minimum_runtime_on_off_time():
    Resource.clearResources()
    device_manager = DeviceManager(max_power_budget=800, debug=True)

    device = DeviceResource("fake_device", 2000)

    policy1 = MinimumRuntime(device, 1)

    device.set_runtime_policy([policy1])

    device.run()

    device_manager.process_managed_devices()

    assert device.running()
    assert policy1.last_time is not None

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

    device_manager.process_managed_devices()

    assert policy1.current_runtime > 1

    device.stop()

    device_manager.process_managed_devices()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(5))

    device_manager.process_managed_devices()

    # We stopped at around 2 seconds of runtime.
    assert policy1.current_runtime < 5


def test_time_of_use_policy():
    Resource.clearResources()

    from aizero.time_of_use_resource import Modes

    dm = DeviceManager(max_power_budget=800, debug=True)

    fake_time_of_use = Resource("TimeOfUse", variables=["mode"])
    fake_time_of_use.set_value("mode", Modes.on_peak)

    device = DeviceResource("fake_device", 100)
    device.set_runtime_policy([TimeOfUsePolicy(Modes.off_peak)])

    device.run()

    assert device.running()

    dm.process_managed_devices()

    assert not device.running()

    fake_time_of_use.set_value("mode", Modes.off_peak)

    dm.process_managed_devices()

    assert device.running()


def test_manager_multiple_power_sources():

    Resource.clearResources()

    ps1 = Resource("PowerSource1", ["current_power"])
    ps2 = Resource("PowerSource2", ["current_power"])

    dm = DeviceManager(power_source=[ps1.name, ps2.name])

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    ps1.set_value("current_power", 10)
    ps2.set_value("current_power", 5)

    assert dm.available_power == 15

    ps1.set_value("current_power", 0)
    ps2.set_value("current_power", 0)

    assert dm.available_power == 0


def test_manager_default_power_source():
    Resource.clearResources()

    ps1 = Resource("SolarPower", ["current_power"])
    ps1.set_value("current_power", 0)

    dm = DeviceManager()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    assert dm.available_power == 0

    ps1.set_value("current_power", 1000)

    assert dm.available_power == 1000


if __name__ == "__main__":
    test_main()
    test_occupancy_policy()
