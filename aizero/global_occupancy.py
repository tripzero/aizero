""" global_occupancy.py

"""
import asyncio
from numpy import float64
from .resource import Resource, MINS


class GlobalOccupancy(Resource):

    def __init__(self):
        super().__init__("GlobalOccupancy", variables=["occupancy"])

        self.occupancy = False

    @asyncio.coroutine
    def poll(self):
        while True:
            self.occupancy = False

            for rsrc in Resource.resources:
                if ("occupancy" in rsrc.variables.keys() and
                        rsrc.name != self.name):

                    occupancy = rsrc.get_value("occupancy")
                    if occupancy is not None:
                        self.occupancy |= occupancy

            self.set_value("occupancy", float64(self.occupancy))

            yield from asyncio.sleep(MINS(3))
