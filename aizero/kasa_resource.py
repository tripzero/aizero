import asyncio

from pyHS100 import Discover
from pyHS100.smartdevice import SmartDeviceException
from pyHS100.smartstrip import SmartStrip
from pyHS100.protocol import TPLinkSmartHomeProtocol
from aizero.device_resource import DeviceResource
from aizero.resource import MINS, has_resource, ResourceNameAlreadyTaken
from aizero.utils import run_thread

TPLinkSmartHomeProtocol.DEFAULT_TIMEOUT = 1


@asyncio.coroutine
def discover_devices():

    devices = yield from run_thread(Discover.discover)

    plugs = []

    for key, device in devices.items():

        try:
            if not has_resource(device.alias):
                if isinstance(device, SmartStrip):
                    for i in range(device.num_children):

                        try:
                            plugs.append(
                                KasaStripPort(device=device,
                                              plug_index=i))
                        except ResourceNameAlreadyTaken:
                            pass

                else:
                    try:
                        plugs.append(
                            KasaPlug(device.alias,
                                     device_name=device.alias,
                                     device=device))
                    except ResourceNameAlreadyTaken:
                        pass
        except SmartDeviceException:
            print("SmartDeviceException with {}".format(
                device.host))
            continue

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
            self.device = yield from run_thread(get_kasa_plug,
                                                self.device_name)
            yield from asyncio.sleep(MINS(1))

        self.has_emeter = self.device.has_emeter
        self.is_dimmable = self.device.is_dimmable

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

    def get_is_on(self):
        return self.device.is_on

    @asyncio.coroutine
    def update(self):
        if not self.device:
            return

        is_on = yield from run_thread(self.get_is_on)

        if is_on and not super().running():
            super().run()

        elif not is_on and super().running():
            super().stop()

        self._update_power_usage()

    def _update_power_usage(self):

        if self.has_emeter:
            consumption = self.device.current_consumption()
            self.update_power_usage(consumption)

    def run(self):
        self.set(True)
        return super().run()

    def stop(self):
        if super().stop():
            self.set(False)

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
                yield from self.update()

            except SmartDeviceException as ex:
                print("KasaPlug error ({}). trying to reconnect".format(
                    self.name))
                print(ex)
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


class KasaStripPort(KasaPlug):

    def __init__(self, device, plug_index):

        name = device.get_sysinfo()["children"][plug_index]["alias"]

        super().__init__(name=name,
                         device_name=device.alias,
                         device=device)

        self.plug_index = plug_index

    def set(self, val):
        """
        Turn on or off the device

        :param val True or False
        """
        if not self.device:
            return

        if val:
            self.device.turn_on(index=self.plug_index)
        else:
            self.device.turn_off(index=self.plug_index)

    @asyncio.coroutine
    def update(self):

        if not self.device:
            return

        is_on = yield from run_thread(self.device.is_on)
        is_on = is_on[self.plug_index]

        if is_on and not super().running():
            super().run()

        elif not is_on and super().running():
            super().stop()

        consumption = yield from run_thread(self.device.current_consumption,
                                            index=self.plug_index)
        self.update_power_usage(consumption)


def test_power_usage_setter():

    plug = KasaPlug("foo", "bar")
    plug.update_power_usage(100)

    assert plug.power_usage == 100


def main():
    from device_resource import DeviceManager
    DeviceManager()

    loop = asyncio.get_event_loop()
    plugs = loop.run_until_complete(discover_devices())

    for plug in plugs:
        print("found: {}".format(plug.name))

    @asyncio.coroutine
    def do_stuff(plugs):

        while True:
            print("number of plugs: {}".format(len(plugs)))
            for plug in plugs:
                print("plug {} is running: {}".format(
                    plug.name, plug.running()))

                print("power: {}".format(plug.power_usage))

                try:
                    print(plug.device.plugs)
                except Exception:
                    pass

            yield from asyncio.sleep(MINS(1))

    loop.run_until_complete(do_stuff(plugs))


if __name__ == "__main__":
    main()
