
import asyncio
from aizero.device_resource import DeviceManager
from aizero.kasa_resource import KasaPlug
from aizero.resource import MINS
from pyHS100 import Discover


def main():
    """
    set power budget to 3000 Watts.  This really doesn't matter because
    all devices have the priority and are not "managed".
    """
    dev_man = DeviceManager(max_power_budget=3000)

    devices = Discover.discover()

    plugs = []

    for key, device in devices.items():
        plug = KasaPlug(device.alias, device=device)
        plugs.append(plug)

        plug.set(True)

    @asyncio.coroutine
    def do_stuff(plugs):

        while True:
            for plug in plugs:
                print("plug {} is running: {} {}W".format(
                    plug.name, plug.running(), plug.power_usage))

            yield from asyncio.sleep(MINS(1))

    asyncio.get_event_loop().run_until_complete(do_stuff(plugs))


if __name__ == "__main__":
    main()
