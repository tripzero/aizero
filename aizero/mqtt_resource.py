# mqtt_resource.py
#
#
#
import asyncio
from functools import partial
from aizero.resource import Resource, ResourceNotFoundException

from gmqtt.client import Client
from gmqtt.mqtt.constants import MQTTv311
from gmqtt.mqtt.handler import MQTTConnectError

import json


class MqttTopicConverter:
    topic_conversion_map = {}

    @staticmethod
    def add_conversion(topic, fnc):
        print("adding {} to topic {}".format(fnc.__name__, topic))
        MqttTopicConverter.topic_conversion_map[topic] = fnc

    @staticmethod
    def has_conversion(topic):
        return topic in MqttTopicConverter.topic_conversion_map

    @staticmethod
    def convert(topic, value):
        return MqttTopicConverter.topic_conversion_map[topic](value)


def mqtt_topic_converter(topic):

    def the_real_decorator(some_function):
        MqttTopicConverter.add_conversion(topic, some_function)

        def wrapper(*args, **kwargs):
            return some_function(*args, **kwargs)

        return wrapper

    return the_real_decorator


class MqttCommon:

    def __init__(self, name=None, broker=None, username=None, password=None,
                 mqtt_protocol_version=MQTTv311):
        self.client = Client(client_id=None)
        self.name = name

        if username is not None:
            self.client.set_auth_credentials(username, password)

        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.mqtt_protocol_version = mqtt_protocol_version

        self.connected = False

        asyncio.get_event_loop().create_task(self.do_connect(broker))

    @asyncio.coroutine
    def do_connect(self, broker):
        try:
            yield from self.client.connect(broker, keepalive=60,
                                           version=self.mqtt_protocol_version)
        except MQTTConnectError as ex:
            print("MqttCommon({}) do_connect failed!".format(self.name))
            print(ex.message)

    def on_connect(self, client, flags, rc, properties):
        self.connected = True

    def on_disconnect(self, packet, exc=None):
        print("MqttCommon({}) disconnected".format(self.name))
        self.connected = False

    @asyncio.coroutine
    def wait_until_connected(self):
        while not self.connected:
            yield from asyncio.sleep(0.1)

    def on_message(self, client, topic, payload, qos, properties):
        print("on_message({}: {}) please override me in subclass.".format(
            topic, payload.decode('utf-8')))


""" MqttResource represents a remote mqtt resource
"""


class MqttResource(Resource, MqttCommon):

    def __init__(self, name, broker, variables=None,
                 variable_mqtt_map=None, quiet=True, **kwargs):
        """
            :param variable_mqtt_map - map mqtt messages to local variables.
                   eg. remote_resource =
                            MqttResource("light_switch1",
                                         "some.broker.com",
                                         ["foo"],
                                         {'foo' : 'light_switch1/foo'})

            :param topic_conversion_map - map of mqtt topics and
                   conversion functions
        """

        if variables is None:
            variables = []

        if variable_mqtt_map is None:
            variable_mqtt_map = {}

        self.quiet = quiet

        Resource.__init__(self, name, variables, ignore_same_value=False)
        MqttCommon.__init__(self, name, broker=broker, **kwargs)

        self.variable_mqtt_map = variable_mqtt_map

    def on_connect(self, client, flags, rc, properties):
        MqttCommon.on_connect(self, client, flags, rc, properties)

        if not self.quiet:
            print("MqttResource connected ({})".format(self.name))

        for variable in self.variables:
            self.subscribe_mqtt(variable)

    def on_message(self, client, topic, payload, qos, properties):

        payload = payload.decode('utf-8')

        if not self.quiet:
            print("MqttResource message: {} : {}".format(topic, payload))

        try:
            if self.variable_mqtt_map is not None:
                for key, val in self.variable_mqtt_map.items():

                    if val == topic:

                        if MqttTopicConverter.has_conversion(topic):
                            payload = MqttTopicConverter.convert(
                                topic, payload)

                        self.setValue(key, payload)
                        return

                    elif key == topic:
                        if MqttTopicConverter.has_conversion(key):
                            payload = MqttTopicConverter.convert(key, payload)

                        self.setValue(key, payload)
                        return

                    elif key == topic:
                        if MqttTopicConverter.has_conversion(key):
                            payload = MqttTopicConverter.convert(key, payload)

                        self.setValue(key, payload)
                        return

            print("topic not found in subscriptions list")
            print("subscriptions: {}".format(self.variable_mqtt_map.keys()))

        except:
            print("exploded in MqttResource.on_message")
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=12, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=12, file=sys.stdout)

    def subscribe_mqtt(self, name):
        if name in self.variable_mqtt_map:
            print("{} is mapped to mqtt topic: {}".format(
                name, self.variable_mqtt_map[name]))
            name = self.variable_mqtt_map[name]

        elif name not in self.variable_mqtt_map:
            self.variable_mqtt_map[name] = name

        print("subscribing to {}".format(name))
        self.client.subscribe(name)


class MqttWrapper(MqttCommon):
    """
    Automatically exports resource properties to to mqtt network
    """

    def __init__(self, resource, broker, override_name=None, retain_msgs=False,
                 qos=0, whitelist_variables=None, blacklist_variables=None,
                 quiet=True, **kwargs):

        super().__init__(name=override_name, broker=broker, **kwargs)

        self.resource = resource
        self.whitelist_variables = whitelist_variables
        self.blacklist_variables = blacklist_variables
        self.retain = retain_msgs
        self.qos = qos
        self.override_name = override_name
        self.quiet = quiet

        if isinstance(resource, str):
            try:
                self.resource = Resource.resource(resource)
                self._subscribe()
            except ResourceNotFoundException:
                def wait_resource():
                    self.resource = Resource.resource(resource)
                    self._subscribe()

                Resource.waitResource(resource, wait_resource)

        elif isinstance(resource, Resource):
            self._subscribe()

        else:
            raise ValueError("Invalid resource type: {}".format(
                resource.__class__.__name__))

        self.whitelist_variables = whitelist_variables
        self.blacklist_variables = blacklist_variables
        self.retain = retain_msgs
        self.qos = qos
        self.override_name = override_name

        self.connected = False

    def on_connect(self, client, flags, rc, properties):
        MqttCommon.on_connect(self, client, flags, rc, properties)

        for variable, value in self.resource.variables.items():
            if value is not None:
                self.publish(variable, value)

    def _subscribe(self):
        # We are guaranteed a valid resource here

        if self.retain is False:
            # We need to publish even if values are the same so that clients
            # get proper value
            self.resource.ignore_same_value = False

        for variable, value in self.resource.variables.items():
            print("MqttWrapper subscribing to: {}".format(variable))
            self.resource.subscribe(variable, partial(self.publish, variable))

            # if there are initial values, publish them
            if value is not None:
                self.publish(variable, value)

    @asyncio.coroutine
    def do_async_publish(self, *args, **kwargs):
        yield from self.wait_until_connected()

        self.client.publish(*args, **kwargs)

    def publish(self, variable, value):
        if (self.whitelist_variables is not
                None and variable not in self.whitelist_variables):
            return

        if (self.blacklist_variables is not
                None and variable in self.blacklist_variables):
            return

        name = self.resource.name

        if self.override_name is not None:
            name = self.override_name

        variable = "{}/{}".format(name, variable)

        if not self.quiet:
            print("MqttWrapper publishing to {}: {}".format(variable, value))

        if isinstance(value, dict):
            if not self.quiet:
                print("MqttWrapper converting dict value {} to json".format(
                    variable))

            value = json.dumps(value)

        if self.connected:
            self.client.publish(variable, "{}".format(
                value), qos=self.qos, retain=self.retain)
        else:
            asyncio.get_event_loop().create_task(
                self.do_async_publish(variable, "{}".format(value),
                                      qos=self.qos, retain=self.retain))


def test_mqtt_wrapper():

    # we need to do this because we will be creating the resources
    # again with the same names. clear from previous tests...
    Resource.clearResources()

    import time

    broker = "127.0.0.1"

    resource = MqttResource("Test1Resource", broker, ["Test1Resource2/foo"])

    publisher = MqttWrapper(
        Resource("Test1Resource2", ["foo"]), broker, retain_msgs=False)

    loop = asyncio.get_event_loop()

    loop.run_until_complete(resource.wait_until_connected())
    loop.run_until_complete(publisher.wait_until_connected())

    publisher.resource.setValue("foo", "bar")

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    assert resource.getValue("Test1Resource2/foo") == "bar"

    publisher.resource.setValue("foo", "bar2")

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    assert resource.getValue("Test1Resource2/foo") == "bar2"


def test_basic_topic_converter():
    MqttTopicConverter.add_conversion("foo", lambda v: int(v))

    assert MqttTopicConverter.convert("foo", "2") == 2


def test_topic_converter():

    # we need to do this because we will be creating the resources
    # again with the same names clear from previous tests...
    Resource.clearResources()

    import time

    broker = "127.0.0.1"

    @mqtt_topic_converter("ResourcePublisher/foo")
    def convert_int(value):
        return int(value)

    publisher = MqttWrapper(Resource("ResourcePublisher", [
                            "foo"]), broker, retain_msgs=False)

    resource = MqttResource("Test2Resource", broker,
                            ["foo"],
                            variable_mqtt_map={
                                "foo": "ResourcePublisher/foo"})

    loop = asyncio.get_event_loop()

    loop.run_until_complete(resource.wait_until_connected())
    loop.run_until_complete(publisher.wait_until_connected())

    publisher.resource.setValue("foo", 1)

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    assert resource.getValue("foo") == 1


def test_multi_decorator_topic_converter():
    import time
    import json

    Resource.clearResources()

    broker = "127.0.0.1"

    @mqtt_topic_converter("Test1Resource2/foo")
    @mqtt_topic_converter("Test1Resource3/baz")
    def convert_int(value):
        return int(value)

    @mqtt_topic_converter("Test1Resource4/some_json")
    def convert_json(value):
        return json.loads(value)

    # we need to do this because we will be creating the resources
    # again with the same names clear from previous tests...
    Resource.clearResources()

    publisher = MqttWrapper(
        Resource("Test1Resource2", ["foo"]), broker, retain_msgs=False)
    publisher2 = MqttWrapper(
        Resource("Test1Resource3", ["baz"]), broker, retain_msgs=False)
    publisher3 = MqttWrapper(
        Resource("Test1Resource4", ["some_json"]), broker, retain_msgs=False)

    resource = MqttResource("Test1Resource", broker,
                            ["foo", "baz", "some_json"],
                            variable_mqtt_map={"foo": "Test1Resource2/foo",
                                               "baz": "Test1Resource3/baz",
                                               "some_json":
                                               "Test1Resource4/some_json"})

    loop = asyncio.get_event_loop()

    loop.run_until_complete(publisher.wait_until_connected())
    loop.run_until_complete(publisher2.wait_until_connected())
    loop.run_until_complete(publisher3.wait_until_connected())
    loop.run_until_complete(resource.wait_until_connected())

    publisher.resource.setValue("foo", 1)
    publisher2.resource.setValue("baz", 1)
    publisher3.resource.setValue("some_json", '{"foo" : 1}')

    loop.run_until_complete(asyncio.sleep(5))

    assert resource.getValue("foo") == 1
    assert resource.getValue("baz") == 1
    assert resource.getValue("some_json")["foo"] == 1

    # Test dict object auto-json conversion:
    publisher3.resource.setValue("some_json", {"foo": 2})

    loop.run_until_complete(asyncio.sleep(5))

    assert resource.getValue("some_json")["foo"] == 2


def test_mqtt_client():

    import time

    broker = "127.0.0.1"

    resource = MqttResource("IAmGroot", broker=broker,
                            variables=["$SYS/broker/uptime"])

    asyncio.get_event_loop().run_until_complete(asyncio.sleep(10))

    assert resource.getValue("$SYS/broker/uptime") is not None


def test_delayed_publish():

    publisher = Resource("publisher", ["foo"])
    publisher.export_mqtt()

    subscriber = MqttResource("publisher_sub", "localhost", ["foo"],
                              variable_mqtt_map={"foo": "publisher/foo"})

    publisher.setValue("foo", "bar")

    assert subscriber.getValue("foo") is None

    loop = asyncio.get_event_loop()

    loop.run_until_complete(publisher.mqtt_wrapper.wait_until_connected())

    assert subscriber.getValue("foo") == "bar"


def main():
    test_mqtt_client()
    test_mqtt_wrapper()


if __name__ == "__main__":
    main()
