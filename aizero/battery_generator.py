from aizero.device_resource import DeviceResource, LinkDevicePolicy
from aizero.resource import ResourceRequires


class BatteryGenerator(DeviceResource):
    """
    BatteryGenerator is a special device that generates power.


    This device is for power generation only. It is intended to be used
    as a power source for the DeviceManager.  Charging device is intended
    to be a separate device and managed separately.
    """

    def __init__(self, name, power_generation, power_control_device,
                 battery_capacity_resource, battery_capacity_min=10,
                 variables=None, **kwargs):
        """
        :param name: The name of the battery generator
        :param power_generation: amount of power that the battery generates
            when in discharge mode.
        :param power_control_device: device that controls power generation.
            Set policies on this device, not the power generator sub-device.
        :param battery_capacity_resource: resource for battery capacity. Must
            have a 'capacity' property as percentage.
        :param variables: custom variables for this resource
        """

        if variables is None:
            variables = []

        if "current_power" not in variables:
            variables.append("current_power")

        if "available_power" not in variables:
            variables.append("available_power")

        super().__init__(name, variables=variables, **kwargs)

        self.available_power = power_generation
        self.battery_capacity_min = battery_capacity_min

        self.ready = False

        def init_generator(rsrcs):

            self.power_control_device = rsrcs(power_control_device)

            self.battery_capacity_resource = rsrcs(battery_capacity_resource)

            assert self.battery_capacity_resource.has_property("capacity"), \
                "Battery capacity resource should have 'capacity' property."

            assert self.power_control_device.runtime_policy == [], \
                "Power control device should not have any runtime policies."

            self.device_manager.ignore_power_usage(self.power_control_device)

            self.bind_from("power_usage",
                           self.power_control_device,
                           "current_power")

            self.set_value("running",
                           self.power_control_device.get_value("running"))

            self.bind_from("running",
                           self.power_control_device)

            self.power_control_device.subscribe(
                "running", self.running_changed)

            if self.running():
                self.set_value("available_power", self.available_power)
            else:
                self.set_value("available_power", 0)

            self.ready = True

        self.rsrcs = ResourceRequires([power_control_device,
                                       battery_capacity_resource],
                                      init_generator)

    def running_changed(self, val):
        if not val:
            self.set_value("available_power", 0)
        else:
            self.set_value("available_power", self.available_power)

    def running(self):
        return self.power_control_device.running()

    def run(self):

        print("******************")
        print("calling BatteryGenerator run()")
        print("******************")

        if not self.ready:
            print("BatteryGenerator is not ready to run.")
            return False

        capacity = self.battery_capacity_resource.get_value("capacity")

        if capacity < self.battery_capacity_min:
            print("BatteryGenerator: capacity too low")
            return False

        if not self.power_control_device.run():
            print("BatteryGenerator: control device failed to run...")
            return False

        self.set_value("available_power", self.available_power)

        return super().run()

    def stop(self):

        print("******************")
        print("calling BatteryGenerator stop()")
        print("******************")

        if not self.power_control_device.running():
            return super().stop()

        if self.power_control_device.stop() and super().stop():
            self.set_value("available_power", 0)
            return True

        return False


def test_battery_generator():
    import asyncio
    from aizero.device_resource import DeviceManager, DeviceResource
    from aizero.resource import Resource

    device_manager = DeviceManager(power_source="BatteryGenerator")

    fake_capacity = Resource("BatteryCapacity", ["capacity"])

    fake_capacity.set_value("capacity", 0)

    fake_battery_control = DeviceResource("FakeControl")

    generator = BatteryGenerator(
        "BatteryGenerator",
        power_generation=1000,
        power_control_device=fake_battery_control.name,
        battery_capacity_resource=fake_capacity.name
    )

    # give manager a chance to grab the power source
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    assert not generator.run()

    device_manager.process_managed_devices()

    assert not fake_battery_control.running()

    fake_capacity.set_value("capacity", 50)

    assert generator.run()
    assert fake_battery_control.running()

    assert generator.get_value("available_power") == 1000

    assert generator.stop()
    assert generator.get_value("available_power") == 0


def test_battery_generator_policy():
    import asyncio
    from aizero.device_resource import DeviceManager, DeviceResource
    from aizero.device_resource import TimeOfUsePolicy
    from aizero.time_of_use_resource import Modes
    from aizero.resource import Resource

    Resource.clearResources()

    fake_time_of_use = Resource("TimeOfUse", variables=["mode"])
    fake_time_of_use.set_value("mode", Modes.on_peak)

    fake_power_device = DeviceResource("FakeGeneratorPowerControl")

    fake_capacity = Resource("BatteryCapacity", ["capacity"])

    fake_capacity.set_value("capacity", 100)

    device_manager = DeviceManager(power_source="BatteryGenerator")

    generator = BatteryGenerator(
        "BatteryGenerator",
        power_generation=1000,
        power_control_device=fake_power_device.name,
        battery_capacity_resource=fake_capacity.name
    )

    generator.set_runtime_policy([TimeOfUsePolicy(Modes.off_peak)])

    # give manager a chance to grab the power source
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

    fake_power_device.stop()
    assert not generator.running()

    fake_time_of_use.set_value("mode", Modes.off_peak)

    device_manager.process_managed_devices()

    assert generator.running()

    fake_time_of_use.set_value("mode", Modes.on_peak)

    device_manager.process_managed_devices()

    assert not generator.running()

    fake_power_device.run()
    assert generator.running()

    fake_power_device.stop()
    assert not generator.running()
