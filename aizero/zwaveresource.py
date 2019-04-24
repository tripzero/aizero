import asyncio

import openzwave
from openzwave.node import ZWaveNode
from openzwave.value import ZWaveValue
from openzwave.scene import ZWaveScene
from openzwave.controller import ZWaveController
from openzwave.network import ZWaveNetwork
from openzwave.option import ZWaveOption
from pydispatch import dispatcher

from aizero.resource import Resource
from aizero.device_resource import DeviceResource

"""
if using docker, here is a good command to get this resource up and running:

    docker --net=host -v/etc/openzwave:/etc/openzwave \
           -v/dev/ttyACM1:/dev/ttyACM1 \
           aizero python3 zwaveresource.py
"""


class Network:

    started = "started"
    ready = "ready"
    failed = "failed"

    def __init__(self, device, logging=False):

        options = ZWaveOption(device,
                              config_path="/etc/openzwave",
                              user_path=".", cmd_line="")

        options.set_log_file("OZW_Log.log")
        options.set_append_log_file(False)
        options.set_console_output(False)
        options.set_logging(logging)
        options.lock()

        dispatcher.connect(
            self.on_started, ZWaveNetwork.SIGNAL_NETWORK_STARTED)
        dispatcher.connect(self.on_failed, ZWaveNetwork.SIGNAL_NETWORK_FAILED)
        dispatcher.connect(self.on_ready, ZWaveNetwork.SIGNAL_NETWORK_READY)

        self.state = None
        self.needs_update = False
        self.resources = []

        self.network = ZWaveNetwork(options, autostart=True)

        asyncio.get_event_loop().create_task(self._loop())

    def create_resource_representation_of_nodes(self):
        print("self.network.nodes.values(): {}".format(
            self.network.nodes.values()))

        for node in self.network.nodes.values():
            print("creating resource for node: {}".format(node))
            try:
                # Locks:
                print("has doorlocks? {}".format(len(node.get_doorlocks())))

                if len(node.get_doorlocks()) > 0:
                    print("{} has doorlocks".format(node))
                    self.resources.append(ZWaveDoorLock(node))

                # Sensors
                else:
                    self.resources.append(Node(node))
            except Exception as ex:
                print("failed to add resource for node {}: {}".format(
                    node, ex))
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=12, file=sys.stdout)

    def resource(self, name):
        names = []
        for rsrc in self.resources:
            names.append(rsrc.name)
            if rsrc.name == name:
                return rsrc

        print("{} not found. Available zwave resources: {}".format(
            name, "\n".join(names)))

    @asyncio.coroutine
    def _loop(self):
        while self.state != Network.ready:
            yield from asyncio.sleep(1)

        self.create_resource_representation_of_nodes()

        while True:
            if self.needs_update:
                for rsrc in self.resources:
                    rsrc._update()
                self.needs_update = False

            yield from asyncio.sleep(1)

    def on_started(self, network):
        self.state = Network.started

    def on_ready(self, network):
        self.state = Network.ready
        dispatcher.connect(self.on_node_update, ZWaveNetwork.SIGNAL_NODE)
        dispatcher.connect(self.on_node_value_changed,
                           ZWaveNetwork.SIGNAL_VALUE)

    def on_failed(self, network):
        self.state = Network.failed

    def on_node_update(self, network, node):
        pass

    def on_node_value_changed(self, node, value):
        self.needs_update = True


class Node(Resource):

    def __init__(self, node):
        self.node = node

        name = node.name

        if node.name == "":
            name = "unknown_zwave_node_id_{}".format(node.node_id)

        variables = []

        print("{} command classes: {}".format(
            name, node.command_classes_as_string))

        print("{} command classes: {}".format(
            name, node.command_classes))

        for value in node.values.values():
            print("resource '{}' creating variable for {} ({})".format(
                  name, value.label, value.command_class))
            print("value type: {}".format(value.type))
            print("value min: {}".format(value.min))
            print("value max: {}".format(value.max))
            print("value help: {}".format(value.help))
            if value.type == "list":
                print("value items: {}".format(value.items))
            variables.append(value.label)

        super().__init__(name=name, variables=variables)

        self._update()

    def _update(self):
        """ This is called by the Network """
        for value in self.node.values.values():
            self.setValue(value.label, value.data)


class DeviceNode(DeviceResource, Node):

    def __init__(self, node):
        name = node.name

        Node.__init__(self, node)
        DeviceResource.__init__(self, name)


class ZWaveDoorLock(Node):

    def __init__(self, node):
        super().__init__(node)
        self.locks = self.node.get_doorlocks()

    def set_lock(self, val):
        for lock in self.locks:
            self.node.set_doorlock(lock, val)

    def get_logs(self):
        return self.node.get_doorlock_logs()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', dest="device",
                        help="specify device ie '/dev/ttyACM0'.",
                        default="/dev/ttyACM0")
    args = parser.parse_args()

    network = Network(args.device, logging=False)

    @asyncio.coroutine
    def poll_loop():
        while True:
            yield from asyncio.sleep(10)

            print("network: {}".format(network.state))

            for rsrc in network.resources:
                print("-------------------")
                print(rsrc.name)
                print("-------------------")
                # print(rsrc.variables)

                if isinstance(rsrc, ZWaveDoorLock):
                    for key, log in rsrc.get_logs().items():
                        print("lock log: {}".format(log.data_as_string))

    asyncio.get_event_loop().run_until_complete(poll_loop())


if __name__ == "__main__":
    main()
