""" global_occupancy.py

"""
import asyncio
from numpy import float64
from .resource import Resource, MINS
from .sys_time import get_current_datetime


class GlobalOccupancy(Resource):

    def __init__(self):
        super().__init__("GlobalOccupancy", variables=["occupancy"])

        self.occupancy = False

    @asyncio.coroutine
    def poll(self):
        while True:
            self.occupancy = False

            print("occupancy check: {}".format(get_current_datetime()))

            for rsrc in Resource.resources:
                if ("occupancy" in rsrc.variables.keys() and
                        rsrc.name != self.name):

                    occupancy = rsrc.get_value("occupancy")

                    print("{} occupancy: {}".format(rsrc.name, occupancy))

                    if occupancy is not None:
                        self.occupancy |= occupancy

            self.set_value("occupancy", float64(self.occupancy))

            yield from asyncio.sleep(MINS(3))
