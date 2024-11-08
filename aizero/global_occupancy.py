""" global_occupancy.py

"""
import asyncio
from numpy import float64
from .resource import Resource, MINS, get_resource
from .sys_time import get_current_datetime


def get_global_occupancy_reader(broker="192.168.1.40"):
    from aizero.mqtt_resource import MqttResource

    return get_resource("GlobalOccupancyReader",
                        MqttResource,
                        name="GlobalOccupancyReader",
                        broker=broker,
                        variables=["occupancy"],
                        variable_mqtt_map={
                            "occupancy": "GlobalOccupancy/occupancy",
                        })


class GlobalOccupancy(Resource):

    def __init__(self):
        super().__init__("GlobalOccupancy", variables=["occupancy"])

        self.occupancy = False

    async def poll(self):
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

            await asyncio.sleep(MINS(3))
