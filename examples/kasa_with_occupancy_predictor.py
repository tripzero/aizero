"""
Turn off kasa devices if the predicted occupancy is False

This example uses a fake occupancy device, but could easily use a real one.

"""

import asyncio
import random

from aizero.device_resource import DeviceManager, OffIfUnoccupied
from aizero.kasa_resource import discover_devices
from aizero.occupancy_predictor import OccupancyPredictorResource
from aizero.resource import Resource


def main():

    # setup a fake occupancy sensor. Normally, this would come from an ecobee
    # device or from a camera resource... anything that has an "occupancy"
    # property.

    fake_occupancy = Resource("Office Occupancy Sensor", [
                              "occupancy"])

    # generate lots of fake data:
    for i in range(100):
        occupied = random.choice([True, False])
        fake_occupancy.set_value("occupancy", occupied)

    # We don't have a power source, so we'll just set the max budget manually:
    DeviceManager(max_power_budget=1000)

    # our predictor will use train and run using the "GlobalOccupancy" resource
    occupancy_predictor = OccupancyPredictorResource(
        name="FakeOccupancyPredictor",
        occupancy_resource=fake_occupancy.name)

    kasa_devices = discover_devices()

    for device in kasa_devices:
        policies = [
            OffIfUnoccupied(occupancy_predictor.name)
        ]

        device.set_runtime_policy(policies)

    # run the mainloop
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
