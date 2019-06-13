import asyncio

from pyHS100 import Discover
from pyHS100.smartdevice import SmartDeviceException

from aizero.device_resource import DeviceResource
from aizero.resource import MINS


def discover_devices():

    devices = Discover.discover()

    plugs = []

    for key, device in devices.items():
        plugs.append(
            KasaPlug(device.alias,
                     device_name=device.alias,
                     device=device))

    return plugs


def get_kasa_device(alias, devices):

    for device in devices:
        if device.device_name == alias:
            return device


def get_kasa_plug(alias):
    devices = Discover.discover()

    for key, device in devices.items():
        if device.alias == alias:
            return device


def alias_to_resource_name(alias):
    return alias.replace(" ", "_")


class KasaPlug(DeviceResource):

    def __init__(self, name, device_name=None, device=None, **kwargs):
        """
        :param name - resource name
        :param device_name name of device to find. Must set device_name or
                           device.
        :param device instance of pyHS100 SmartDevice
        :param kwargs - key word arguments same as DeviceResource
        """
        super().__init__(name, **kwargs)

        self.device_name = device_name
        self.device = device

        if device_name is None and device is None:
            raise Exception("KasaPlug must have device_name or device set")

        self.has_emeter = False
        self.is_dimmable = False

        asyncio.get_event_loop().create_task(self.process())

        if device_name is not None and device is None:
            asyncio.get_event_loop().create_task(self.do_get_kasa_plug())

        if device is not None:
            self.has_emeter = self.device.has_emeter
            self.is_dimmable = self.device.is_dimmable

    @asyncio.coroutine
    def do_get_kasa_plug(self):
        self.device = None

        while not self.device:
            self.device = get_kasa_plug(self.device_name)
            yield from asyncio.sleep(MINS(1))

        self.has_emeter = self.device.has_emeter
        self.is_dimmable = self.device.is_dimmable

    def running(self):
        return super().running() and self.device and self.device.is_on

    def set(self, val):
        """
        Turn on or off the device

        :param val True or False
        """
        if not self.device:
            return

        if val:
            self.device.turn_on()
        else:
            self.device.turn_off()

    def update(self):
        if not self.device:
            return

        is_on = self.device.is_on

        if is_on and not super().running():
            super().run()

        elif not is_on and super().running():
            super().stop()

        if self.has_emeter:
            emeter_status = self.device.get_emeter_realtime()
            self.update_power_usage(emeter_status['power'])

    def run(self):
        self.set(True)
        self.update()

    def stop(self):
        if super().stop():
            self.set(False)

    @property
    def power_usage(self):
        if self.has_emeter:
            return self.device.current_consumption()

        return super().power_usage

    @property
    def dimmer_level(self):
        if self.is_dimmable:
            return self.device.brightness

        return 0

    @dimmer_level.setter
    def dimmer_level(self, dim_level):
        try:
            if self.is_dimmable and self.running():
                self.device.brightness = dim_level
        except SmartDeviceException:
            pass

    @asyncio.coroutine
    def process(self):
        while True:
            try:
                self.update()
            except SmartDeviceException:
                print("KasaPlug error. trying to reconnect")
                yield from self.do_get_kasa_plug()
            except Exception:
                print("error in kasa_resource process()")
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=12, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=12, file=sys.stdout)

            yield from asyncio.sleep(MINS(1))


def test_power_usage_setter():

    plug = KasaPlug("foo", "bar")
    plug.update_power_usage(100)

    assert plug.power_usage == 100


def main():
    from pyHS100 import Discover
    from device_resource import DeviceManager
    DeviceManager()

    plugs = discover_devices()

    @asyncio.coroutine
    def do_stuff(plugs):

        while True:
            for plug in plugs:
                print("plug {} is running: {}".format(
                    plug.name, plug.running()))
                yield from asyncio.sleep(MINS(1))

    asyncio.get_event_loop().run_until_complete(do_stuff(plugs))


if __name__ == "__main__":
    main()
