import numpy as np


from aizero.resource import Resource, ResourceNotFoundException, MINS
from aizero.resource import get_resource as gr
from remoteresource import RemoteRestResource

import asyncio
from resource_py3 import Py3PollWhileTrue as poll_while_true
from resource_py3 import Py3Resource as resource_poll


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

    @classmethod
    def policy(self, policy_name, **kwargs):
        """
        returns policy instance for matching policy_name or None
        """
        pass


class RuntimePolicy:
    """
        accepts a policy and a set of conditions.  conditions are a list of
        callables that return either True or False.  If all the conditions are
        True, run() will be called on the device.  A conditionthat returns
        False will trigger the stop() call on the device.

        A condition can return None, which indicates no action.
    """

    def __init__(self, policy=RuntimePolicies.none, conditions=[]):
        self.policy = policy
        self.conditions = conditions

    def process(self, device):
        pass

    def has_conditions(self):
        """ return if there are conditions or not. if this returns False,
        run_conditions() is not called.
        """
        return len(self.conditions) > 0

    def run_conditions(self):

        ret_val = True
        do_something = False

        for condition in self.conditions:
            cond_ret_val = condition()

            do_something |= cond_ret_val is not None

            if cond_ret_val is not None:
                ret_val &= cond_ret_val

        if do_something:
            return ret_val


class RunIfCanPolicy(RuntimePolicy):

    def __init__(self, conditions=[]):
        RuntimePolicy.__init__(
            self, policy=RuntimePolicies.run_if_can, conditions=conditions)
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

    def __init__(self, dim_level=50, conditions=[]):
        RuntimePolicy.__init__(
            self, policy=RuntimePolicies.dim_if_cannot, conditions=conditions)

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
        Requires "occupancy" property be available on resource
        optional "predicted_occupancy" property will be used if available
    """

    def __init__(self, occupancy_resource_name, prediction_threshold=0.60,
                 conditions=[]):
        super().__init__(policy=RuntimePolicies.off_if_unoccupied,
                         conditions=conditions)

        self.occupancy = None
        self.predicted_occupancy = None
        self.prediction_threshold = prediction_threshold

        def wait_occupancy():
            self.occupancy_resource = gr(occupancy_resource_name)
            self.occupancy = self.occupancy_resource.subscribe2("occupancy")

            if self.occupancy_resource.hasProperty("predicted_occupancy"):
                self.predicted_occupancy = self.occupancy_resource.subscribe2(
                    "predicted_occupancy")

        try:
            gr(occupancy_resource_name)
            wait_occupancy()
        except ResourceNotFoundException:
            Resource.waitResource(occupancy_resource_name, wait_occupancy)

    def _pass_process(self, device):
        pass

    def process(self, device):
        if (self.can_run not in self.conditions and
            self.occupancy is not None and
                self.occupancy.value is not None):

            self.conditions.append(self.can_run)

            # we are ready. no more need for process to do anything
            self.process = self._pass_process

    def can_run(self):
        is_occupied = self.occupancy.value
        # print("occupied: {}".format(is_occupied))

        if (self.predicted_occupancy and
                self.predicted_occupancy.value is not None):

            is_occupied |= (self.predicted_occupancy.value >
                            self.prediction_threshold)

        if not is_occupied:
            return False


class LinkDevicePolicy(RuntimePolicy):
    """
    Links a device with another device.  When the other device is running,
    this will too.
    """

    def __init__(self, linked_device_name, conditions=[]):
        super().__init__(policy=RuntimePolicies.link_device_policy,
                         conditions=conditions)

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


class DeviceManager(Resource):
    def __init__(self, name="DeviceManager", max_power_budget=0,
                 power_source="SolarPower"):
        """ max_power_budget is to be used to define the max power budget when
            there is no solar system (ie battery system, or pure-grid system).

            power_source must have the "current_power" property unless max
            power_budget is set.

            solar_power is max_power_budget until solar system updates.
        """

        super().__init__(name, ["total_power", "running_devices"])

        self.running_devices = []
        self.managed_devices = []

        self.max_power_budget = max_power_budget
        self.solar_power = max_power_budget
        self.time_of_use_mode = None

        def wait_power_source():
            gr(power_source).subscribe(
                "current_power", self.solar_power_changed)

        if not max_power_budget:
            Resource.waitResource(power_source, wait_power_source)

        def wait_time_of_use():
            print("I got time of use resource!")
            gr("TimeOfUse").subscribe("mode", self.time_of_use_changed)

        Resource.waitResource("TimeOfUse", wait_time_of_use)

        self.poller = resource_poll(self.process_managed_devices, MINS(1))

    def solar_power_changed(self, value):
        self.solar_power = value
        self.debug_print_capacity()
        self.process_managed_devices()

    def time_of_use_changed(self, value):
        self.time_of_use_mode = value
        self.process_managed_devices()

    def register_managed_device(self, device):
        if device not in self.managed_devices:
            self.managed_devices.append(device)
            device.subscribe("power_usage", self.device_power_usage_changed)

    @property
    def highest_consuming_managed_device(self):
        pigs = {}

        for device in self.managed_devices:
            pigs[device.power_usage] = device

        return pigs[np.max(list(pigs.keys()))]

    def debug_print_capacity(self):
        print("total budget: {}W".format(self.solar_power))
        print("total usage: {}W".format(round(self.running_power), 1))
        print("capacity: {}%".format(
            round(self.running_power / max(self.solar_power, 1) * 100.0, 1)))

    def process_managed_devices(self):
        for device in self.managed_devices:

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
                    # print("can_run? {}".format(can_run))

            if has_any_conditions and can_run:
                if not device.running():
                    self.debug_print_capacity()
                    print("device {} can run. {} W. calling run()".format(
                        device.name, device.max_power_usage))
                    print("estimated power usage will be: {}W".format(
                        self.running_power + device.max_power_usage))
                    print("\n".join(can_run_policies))
                    device.run()

            elif has_any_conditions and can_run is False:
                if device.running():
                    print("device {} can NOT run. {} W. calling stop()".format(
                        device.name, device.power_usage))
                    self.debug_print_capacity()
                    print("\n".join(can_not_run_policies))
                    device.stop()
                    # self.debug_print_capacity()

    def device_power_usage_changed(self, value):
        self.process_managed_devices()
        # pass

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
        power_delta = self.running_power - self.solar_power

        # if this device is already "running", it's power is already included
        if device_to_run and not self.is_running(device_to_run):
            power_delta = (self.running_power +
                           device_to_run.max_power_usage) - self.solar_power

        if device_to_run:
            return max(0, min(power_delta, device_to_run.max_power_usage))

        return max(0, power_delta)

    def can_run(self, device):
        # print("can_run: device: {}".format(device.name))

        over_budget_amount = self.overrun_amount(device)
        cr = over_budget_amount == 0

        # print("can_run: device under budget: {}".format(cr))
        # print("can_run: over budget amount: {}".format(over_budget_amount))

        is_time_of_use_mode = False
        if self.time_of_use_mode:
            for mode in device.runtime_modes:
                is_time_of_use_mode |= mode == self.time_of_use_mode

        cr |= is_time_of_use_mode

        # print("can_run: is time of use mode: {}".format(is_time_of_use_mode))

        has_kickable = self.kickable_devices(device, over_budget_amount, True)

        # finally, check if there are lower priority devices we can bump:
        cr |= has_kickable is not None

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

    @property
    def running_devices_pretty(self):
        rdp = []
        for device in self.running_devices:
            rdp.append(device.name)

        return rdp


class DeviceResource(Resource):

    def __init__(self, name, power_usage=0, runtime_modes=[], variables=[],
                 priority=None, runtime_policy=[],
                 device_manager="DeviceManager", resource_args={}):
        """ :param power_usage how much power this device consumes in Watts
            :param runtime_modes time-of-use modes this device can run in. if
             runtime_modes == [], device can run in all runtime modes

            :param runtime_policy policies that determines whether this device
             will run or not. if runtime_policy == None, DeviceManager will
             not manage the device
            :param priority the priority of the device. if not set,
             DeviceManager will not manage the device (see RuntimePriority)
        """
        vars = ["power_usage", "runtime_modes", "priority", "running"]

        for variable in variables:
            if variable not in vars:
                vars.append(variable)

        Resource.__init__(self, name, vars, **resource_args)

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

        def register_with_manager():
            if self.runtime_policy:
                self.device_manager.register_managed_device(self)

        try:
            if isinstance(device_manager, DeviceManager):
                self.device_manager = device_manager
            else:
                self.device_manager = Resource.resource(device_manager)

            register_with_manager()

        except ResourceNotFoundException:
            self.device_manager = None

            def wait_device_manager():
                self.device_manager = Resource.resource(device_manager)
                register_with_manager()

            Resource.waitResource(device_manager, wait_device_manager)

    @property
    def runtime_policy(self):
        return self._runtime_policy

    @runtime_policy.setter
    def runtime_policy(self, value):
        if not isinstance(value, list):
            self._runtime_policy = [value]
        else:
            self._runtime_policy = value

        if self.device_manager:
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

    def wait_can_run(self, callback=None):
        def wait_func():

            if not self.can_run():
                return True

            self.run()

            if callback:
                callback()

            return False

        poll_while_true(wait_func, 1)

    def can_run(self):
        return self.device_manager and self.device_manager.can_run(self)

    def request_run(self):
        """
        request that this device run. Returns request id to be used to remove
        the request.
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
        if not self.device_manager:
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


class RemoteRestDeviceResource(RemoteRestResource, DeviceResource):

    """
        variable_map = {"local_variable_name" : "remote_variable_name"}
    """

    def __init__(self, name, hammock_instance, variable_map={}, **kwargs):

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
    assert kickables != None

    assert device_none not in device_manager.kickable_devices(
        device_high_too_much_power, power_delta)
    assert device_low in device_manager.kickable_devices(
        device_high_too_much_power, power_delta)

    device_high_too_much_power.run()

    assert not device_low.running()
    assert device_high_too_much_power.running()


def test_remote():
    device_manager = DeviceManager(max_power_budget=800)

    from hammock import Hammock

    remote_resource = RemoteRestDeviceResource("GreenhouseDeviceManager", Hammock("https://tripzero.reesfamily12.com:8069/DeviceManager/DeviceManager"),
                                               variable_map={"power_usage": "total_power"}, device_manager=device_manager)

    assert "total_power" in remote_resource.variables

    # run the even loop so "remote_resource.poll" is hit
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    assert remote_resource.getValue(
        "total_power") == remote_resource.getValue("power_usage")


def test_occupancy_policy():
    device_manager = DeviceManager(max_power_budget=800)

    occupancy_resource = Resource(
        "SomeOccupancyThing", variables=["occupancy"])

    occupancy_resource.setValue("occupancy", True)

    off_policy = OffIfUnoccupied(occupancy_resource.name)

    dev1 = DeviceResource("dev1", power_usage=100, device_manager=device_manager,
                          runtime_policy=[off_policy])

    dev1.run()

    device_manager.process_managed_devices()

    assert dev1.running(), "Device should be running"

    occupancy_resource.setValue("occupancy", False)

    assert off_policy.occupancy.value == False

    device_manager.process_managed_devices()

    assert not dev1.running(), "Device should not be running"

    occupancy_resource.setValue("occupancy", True)

    device_manager.process_managed_devices()

    assert not dev1.running(), "Device should not be running"


if __name__ == "__main__":
    test_main()
    test_remote()
    test_occupancy_policy()
