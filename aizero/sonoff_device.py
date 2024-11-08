import asyncio

import sonoff

from aizero.resource import get_resource as gr
from aizero.resource import MINS, HOURS
from aizero.device_resource import DeviceResource
from aizero.utils import run_thread

"""
requires sonoff-python from: https://github.com/lucien2k/sonoff-python
"""


class SonoffDevice(DeviceResource):

    def __init__(self, name, username=None, password=None, region=None):

        super().__init__(name)

        self.username = username
        self.password = password
        self.region = region

        self.update_running = False
        self.device = None
        self.s = None

        asyncio.get_event_loop().create_task(self.full_login())

    async def full_login(self):
        while True:
            if self.username is None:
                self.username = gr(
                    "ConfigurationResource").config["sonoff_username"]
            if self.password is None:
                self.password = gr(
                    "ConfigurationResource").config["sonoff_password"]
            if self.region is None:
                self.region = gr("ConfigurationResource").config.get(
                    "sonoff_region", "us")

            self.s = sonoff.Sonoff(self.username, self.password, self.region)

            await self.update()

            await asyncio.sleep(HOURS(8))

    async def get_device(self):
        for device in self.s.get_devices():
            print(f"searching device {device['name']}")
            if device['name'] == self.name:
                print(f"found device {self.name}")
                self.device = device

    async def poll(self):

        while True:
            await self.update()
            await asyncio.sleep(MINS(3))

    async def update(self):

        if self.s is None:
            return

        if self.update_running:
            while self.update_running:
                await asyncio.sleep(0.1)

            return

        self.update_running = True
        print("running update")

        await run_thread(self.s.do_reconnect)
        await run_thread(self.s.do_login)
        await run_thread(self.s.update_devices)

        await self.get_device()

        is_on = await run_thread(self.get_is_on)

        self.set_value("running", is_on)

        if is_on and not super().running():
            super().run()

        elif not is_on and super().running():
            super().stop()

        self._update_power_usage()
        self.update_running = False

    def get_is_on(self):

        if self.device is None:
            return False

        is_on = self.device['params']['switch']
        is_on = is_on == "on"

        print(f"device {self.name} is_on {is_on}")

        return is_on

    def set(self, val):
        if self.device is None:
            return

        if val:
            val = 'on'
        else:
            val = 'off'

        asyncio.get_event_loop().create_task(self.do_set(val))

    async def do_set(self, val):
        await self.update()
        await run_thread(
            self.s.switch, val, self.device['deviceid'], None)

        await self.poll()

    def run(self):
        self.set(True)
        return super().run()

    def stop(self):
        if super().stop():
            self.set(False)

    def _update_power_usage(self):

        if self.device is not None and "power" in self.device['params']:
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

    if not device.running():
        device.run()
    else:
        device.stop()

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    print(f"device running? {device.running()}")
    print(f"device running? {device.power_usage}")

    if device.running():
        device.stop()
    else:
        device.run()


if __name__ == '__main__':
    main()
