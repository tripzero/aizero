import asyncio
from aiohttp import ClientSession
import pymyq
from pymyq.device import STATE_OPEN
# from pymyq.errors import MyQError

from aizero.resource import MINS
from aizero.device_resource import DeviceResource


class MyQResource(DeviceResource):

    def __init__(self, device_name, username, passwd, brand="liftmaster"):

        super().__init__(name=device_name, variables=["state"])

        self.device = None

        self.username = username
        self.passwd = passwd
        self.brand = brand

        self.trying_to_run = False

        self.poll_rate = MINS(1)

        asyncio.get_event_loop().create_task(self.async_connect())

    @asyncio.coroutine
    def try_to_run(self, run):

        if self.trying_to_run:
            return

        self.trying_to_run = True

        self.set(run)

        yield from asyncio.sleep(self.poll_rate)

        self.trying_to_run = False

    @asyncio.coroutine
    def async_connect(self):

        self.websession = ClientSession()

        myq = yield from pymyq.login(self.username,
                                     self.passwd,
                                     self.brand,
                                     self.websession)

        devices = yield from myq.get_devices()

        device_names = []

        for device in devices:
            device_names.append(device.name)
            if device.name == self.name:
                self.device = device

        if self.device is None:
            print("available devices: {}".format(", ".join(device_names)))
            raise Exception("could not find MyQ device: {}".format(self.name))

    @asyncio.coroutine
    def async_set(self, val):
        if val:
            yield from self.device.open()

        else:
            yield from self.device.close()

    def set(self, val):
        asyncio.get_event_loop().create_task(self.async_set(val))

    def run(self):

        if not super().run():
            return False

        asyncio.get_event_loop().create_task(self.try_to_run(True))

        return True

    def stop(self):

        if not super().stop():
            return False

        asyncio.get_event_loop().create_task(self.try_to_run(False))

        return True

    def running(self):

        return (super().running() and
                self.get_value("state") == STATE_OPEN)

    @asyncio.coroutine
    def poll(self):
        while True:
            if self.device:
                yield from self.device.update()

                self.set_value("state", self.device.state)
                # print("device state: {}".format(self.device.state))

            yield from asyncio.sleep(self.poll_rate)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest="device",
                        help="name of device")
    parser.add_argument("--username", dest="username",
                        help="myQ username/email")
    parser.add_argument("--passwd", dest="passwd",
                        help="myQ password")

    args = parser.parse_args()

    MyQResource(args.device, args.username, args.passwd)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
