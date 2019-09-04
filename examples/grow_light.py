"""
grow_light.py

Find "grow light" devices and make sure they only run when there is available
power.  If there is no available power, the lights will turn off.

This assumes that the plants are lower priority than other managed devices and
that if the lights turn off, no harm will befall the plants.

Go green or go home.

We also want the grow lights to turn off at night so the plants can sleep.

"""
import asyncio

from aizero.device_resource import DeviceManager, RunIfCanPolicy
from aizero.device_resource import DeviceResource
from aizero.resource import Resource
from aizero.nighttime import NightTime


def main():

    DeviceManager(max_power_budget=1000)
    night_time = NightTime().subscribe2("night_time")

    # create a couple of fake devices
    DeviceResource("fake grow light 1", power_usage=100)
    DeviceResource("fake grow light 2", power_usage=100)

    for rsrc in Resource.resources:

        if "grow light" in rsrc.name.lower():
            rsrc.set_runtime_policy([RunIfCanPolicy(conditions=[
                lambda: not night_time.value])])

    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
