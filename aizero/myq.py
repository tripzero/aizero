import asyncio
from aiohttp import ClientSession
import pymyq
# from pymyq.device import STATE_CLOSED, STATE_OPEN
# from pymyq.errors import MyQError

from aizero.resource import Resource, MINS


class MyQResource(Resource):

    def __init__(self, device_name, username, passwd, brand="liftmaster"):

        super().__init__(device_name, ["state"])

        self.device = None

        self.username = username
        self.passwd = passwd
        self.brand = brand

        asyncio.get_event_loop().create_task(self.async_connect())

    @asyncio.coroutine
    def async_connect(self):

        self.websession = ClientSession()

        myq = yield from pymyq.login(self.username,
                                     self.passwd,
                                     self.brand,
                                     self.websession)

        devices = yield from myq.get_devices()

        for device in devices:
            if device.name == self.name:
                self.device = device

        if self.device is None:
            raise Exception("could not find MyQ device: {}".format(self.name))

    @asyncio.coroutine
    def async_set(self, val):
        if val:
            yield from self.device.open()

        else:
            yield from self.device.close()

    def set(self, val):
        asyncio.get_event_loop().create_task(self.async_set(val))

    @asyncio.coroutine
    def poll(self):
        while True:
            if self.device:
                yield from self.device.update()

                self.set_value("state", self.device.state)
                # print("device state: {}".format(self.device.state))

            yield from asyncio.sleep(MINS(1))


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

    print(args)

    MyQResource(args.device, args.username, args.passwd)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
