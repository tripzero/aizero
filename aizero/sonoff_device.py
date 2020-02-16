import asyncio

import sonoff

from aizero.resource import get_resource as gr
from aizero.resource import MINS
from aizero.device_resource import DeviceResource
from aizero.utils import run_thread


class SonoffDevice(DeviceResource):

    def __init__(self, name, username=None, password=None, region=None):

        super().__init__(name)

        if username is None:
            username = gr("ConfigurationResource").config["sonoff_username"]
        if password is None:
            password = gr("ConfigurationResource").config["sonoff_password"]
        if region is None:
            region = gr("ConfigurationResource").config.get(
                "sonoff_region", "us")

        self.s = sonoff.Sonoff(username, password, region)

        self.device = None

        self.get_device()

    def get_device(self):
        for device in self.s.get_devices():
            print(f"searching device {device['name']}")
            if device['name'] == self.name:
                print(f"found device {self.name}")
                self.device = device

    @asyncio.coroutine
    def poll(self):

        while True:
            yield from self.update()
            yield from asyncio.sleep(MINS(3))

    def update(self):
        print("running update")
        yield from run_thread(self.s.update_devices)

        self.get_device()

        is_on = yield from run_thread(self.get_is_on)

        if is_on and not super().running():
            super().run()

        elif not is_on and super().running():
            super().stop()

        self._update_power_usage()

    def get_is_on(self):

        if self.device is None:
            return False

        is_on = self.device['params']['switch']
        is_on = is_on == "on"

        print(f"device is_on {is_on}")

        return is_on

    def set(self, val):
        if self.device is None:
            return

        if val:
            val = 'on'
        else:
            val = 'off'

        asyncio.get_event_loop().create_task(self.do_set(val))

    @asyncio.coroutine
    def do_set(self, val):
        yield from run_thread(
            self.s.switch, val, self.s.device['deviceid'], None)

        yield from self.poll()

    def run(self):
        self.set(True)
        return super().run()

    def stop(self):
        if super().stop():
            self.set(False)

    def _update_power_usage(self):

        if "power" not in self.device['params']:
            consumption = float(self.device['params']['power'])
            self.update_power_usage(consumption)


def main():
    import argparse
    from aizero.device_resource import DeviceManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--username')
    parser.add_argument('--password')

    args = parser.parse_args()

    dev_man = DeviceManager()

    device = SonoffDevice("BatteryGenerator",
                          args.username,
                          args.password,
                          "us")

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    print(f"device running? {device.running()}")
    print(f"device running? {device.power_usage}")


if __name__ == '__main__':
    main()
