import inspect
import json
import numpy as np
import os
import pandas as pd
import sys
import traceback

from datetime import datetime, timezone
from fuzzywuzzy import fuzz

if sys.version_info >= (3, 0):
    import asyncio
    from aizero import Py3PollWhileTrue as poll_while_true
else:
    import trollius as asyncio
    from aizero import Py2PollWhileTrue as poll_while_true

''' This is the basic resource class.  It represents a simple interface to
    subscribe to changes in different properties
'''


def HOURS(hrs):
    return hrs * 3600


def MINS(mins):
    return mins * 60


def DAYS(days):
    return days * HOURS(24)


def get_resource(resource_name, class_=None, **kwargs):
    try:
        return Resource.resource(resource_name)
    except ResourceNotFoundException:
        if class_ is not None:
            return class_(**kwargs)

    raise ResourceNotFoundException(resource_name)


def find_resource(resource_name, threshold=70):
    name = resource_name.lower()
    rsrcs = []
    scores = []

    for r in Resource.resources:

        score = fuzz.token_sort_ratio(
            resource_name, r.name)
        
        if score > threshold:
            # print(f"found {r.name}, {score}")
            rsrcs.append(r)
            scores.append(score)

    if len(rsrcs):
        rsrc = rsrcs[np.argmax(scores)]
        return rsrc


async def wait_for_resource(resource_name):
    while not has_resource(resource_name):
        await asyncio.sleep(1)

    return get_resource(resource_name)


def has_resource(rsrc_name):
    found = rsrc_name in Resource.getResourcesNames()

    return found


def to_timestamp(dt):
    """
    :param dt datetime object
    returns unix timestamp
    """

    return dt.replace(tzinfo=timezone.utc).timestamp()


class ResourceNotFoundException(Exception):
    pass


class ResourceNameAlreadyTaken(Exception):
    pass


class PropertyDoesNotExistException(ValueError):
    pass


class ResourceRequires:

    def __init__(self, required_resources, fulfilled_callback=None):

        self._resources = {}

        if isinstance(required_resources, str):
            required_resources = [required_resources]

        self.required_resources = required_resources
        self.fulfilled_callback = fulfilled_callback

        self._fulfilled = False

        try:
            self.requirements_callback()
        except ResourceNotFoundException:
            Resource.waitResource(self.required_resources,
                                  self.requirements_callback)

    def __call__(self, resource_name):
        return self._resources[resource_name]

    def resources(self):
        return list(self._resources.values())

    def requirements_callback(self):
        for resource in self.required_resources:
            self._resources[resource] = get_resource(resource)

        self._fulfilled = True
        if self.fulfilled_callback is not None:
            self.fulfilled_callback(self)

    @property
    def fulfilled(self):
        return self._fulfilled


class ResourcePropertySubscription:

    def __init__(self, resource, variable):
        self.resource = resource
        self.variable = variable
        self._value = resource.get_value(variable)

    @property
    def value(self):
        return self.resource.get_value(self.variable)

    def __call__(self):
        return self.value


class Resource(object):
    callables = {}
    resources = []
    _auto_export_mqtt = False
    _mqtt_broker = "localhost"
    _mqtt_wrapper_args = None

    def __init__(self, name, variables, loop=None, ignore_same_value=True,
                 mqtt_export_options=None, no_snapshot=False):
        self.name = name
        self.deviceName = name
        self.ignore_same_value = ignore_same_value
        self.data_frame = None
        self.no_snapshot = no_snapshot

        Resource.register(self)

        if not isinstance(variables, list):
            assert ValueError("variables must be a list")

        self.variables = {}
        self.virtualDevice = True
        self.enabled = True

        if not loop:
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = loop

        for v in variables:
            self.variables[v] = None

        self.subscriptions = {}

        self.loop.create_task(self.poll())

        if Resource._auto_export_mqtt:
            export_opts = mqtt_export_options

            if export_opts is None:
                export_opts = Resource._mqtt_wrapper_args

            self.export_mqtt(broker=Resource._mqtt_broker,
                             mqtt_export_options=export_opts)

    @staticmethod
    def auto_export_mqtt(auto_export, broker="localhost",
                         mqtt_export_options=None):
        Resource._auto_export_mqtt = auto_export
        if auto_export:
            Resource._mqtt_broker = broker
            Resource._mqtt_wrapper_args = mqtt_export_options

    def export_mqtt(self, broker="localhost", mqtt_export_options=None):
        from aizero.mqtt_resource import MqttWrapper

        if mqtt_export_options is None:
            mqtt_export_options = {}

        if "broker" not in mqtt_export_options:
            mqtt_export_options["broker"] = broker

        self.mqtt_wrapper = MqttWrapper(self, **mqtt_export_options)

    @property
    def unique_name(self):
        return self.deviceName + "/" + self.name

    @staticmethod
    def register(other):
        if other.name in Resource.getResourcesNames():
            raise ResourceNameAlreadyTaken(
                "Resource {} name already registered".format(other.name))

        Resource.resources.append(other)

    def register_callable_class(self):

        for name, method in inspect.getmembers(self,
                                               predicate=inspect.ismethod):
            if hasattr(method, "is_callable"):
                print("******************************************")
                print("we found a callable on this instance '{}'".format(
                    self.unique_name))
                if self.unique_name not in Resource.callables:
                    Resource.callables[self.unique_name] = {}
                Resource.callables[self.unique_name][name] = method

    def callables(self):
        try:
            return list(Resource.callables[self.unique_name].values())
        except AttributeError as at:
            print(at)
            print("did you forget to call register_callable_class()?")

    @staticmethod
    def resource(name, device=None):
        for r in Resource.resources:
            if r.name == name and (not device or r.deviceName == device):
                return r

        raise ResourceNotFoundException("Resource {} not found".format(name))

    @staticmethod
    def getResourcesNames():
        resource_names = []
        for r in Resource.resources:
            resource_names.append(r.name)

        return resource_names

    @staticmethod
    def clearResources():
        Resource.resources = []

    @staticmethod
    def waitResource(resource_name, callback):
        """resource_name can be a str or list of strings"""

        def wait_resource_temp():
            return Resource._waitResource(resource_name, callback)

        poll_while_true(wait_resource_temp, 1)

    @staticmethod
    def _waitResource(resource_name, callback):
        try:
            if isinstance(resource_name, list):
                for rn in resource_name:
                    if rn is not None:
                        Resource.resource(rn)
            else:
                Resource.resource(resource_name)

            try:
                callback()
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=6, file=sys.stdout)
            # stop looping
            return False
        except Exception:
            pass

        return True

    @classmethod
    def make_callable(sey_whut, func):
        func.is_callable = True
        return func

    @classmethod
    def makeCallable(sey_whut, func):
        func.is_callable = True
        return func

    def subscribe(self, variable, callback):
        if variable not in self.variables:
            raise Exception(
                "Variable {0} does not exist".format(variable))

        if variable not in self.subscriptions:
            # set to empty list
            self.subscriptions[variable] = []

        self.subscriptions[variable].append(callback)
        return True

    def subscribe2(self, property):
        if not self.hasProperty(property):
            raise ValueError("property {} does not exist".format(
                property))

        subscription = ResourcePropertySubscription(self, property)
        return subscription

    def has_property(self, property):
        return property in self.variables

    def hasProperty(self, property):
        """
        deprecated. use has_property
        """
        return self.has_property(property)

    @property
    def properties(self):
        return list(self.variables.keys())

    def propertyChanged(self, property, value):
        if property not in self.subscriptions:
            return

        callbacks = self.subscriptions[property]

        for cb in callbacks:
            try:
                cb(value)
            except Exception:
                print("error calling subscription callback")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=12, file=sys.stdout)

    def setValue(self, property, value):
        """
        deprecated. use set_value
        """
        return self.set_value(property, value)

    def set_value(self, property, value):
        if not self.hasProperty(property):
            raise PropertyDoesNotExistException(
                "Invalid property: {}".format(property))

        if self.ignore_same_value and value == self.variables[property]:
            # print("is same value. bailing")
            return

        self.variables["timestamp"] = to_timestamp(datetime.utcnow())
        self.variables[property] = value

        self.snapshot()

        # print("debug: {} : variable: {}, value: {}, columns: {}".format(
        #   self.name, property, value, self.dataframe.columns))

        self.propertyChanged(property, value)

    def getValue(self, property):
        """
        deprecated. use get_value
        """
        return self.get_value(property)

    def get_value(self, property):
        """
        get the value for a property

        :param: property name of property
        """
        if property not in self.variables:
            print("{} not available in {}".format(property, self.name))
            print("available properties: {}".format(self.variables.keys()))
            raise PropertyDoesNotExistException(
                "Invalid property: {}".format(property))

        return self.variables[property]

    def bind_to(self, property, other_resource, other_property=None):

        if other_property is None:
            other_property = property

        if not self.has_property(property):
            raise ResourceNotFoundException(
                f"{self.name} has no property {property}")

        if not other_resource.has_property(other_property):
            raise ResourceNotFoundException(
                f"{other_resource.name} has no property {other_property}")

        self.subscribe(
            property,
            lambda val: other_resource.set_value(other_property, val))

        other_resource.set_value(other_property, self.get_value(property))

    def bind_from(self, property, other_resource, other_property=None):
        return other_resource.bind_to(property, self, other_property)

    def snapshot(self):
        """
        create/update dataframe from current values
        """

        if self.no_snapshot:
            return

        if self.data_frame is None:
            self.restore(None)

        vars = self.variables.copy()

        for k in vars.keys():
            v = vars[k]
            if v is None:
                vars[k] = np.nan

            # We don't support list types ATM
            elif isinstance(v, list):
                vars[k] = np.nan

        ts = None
        if "timestamp" in self.variables:
            ts = pd.Index([vars.pop("timestamp")], name="timestamp")

        df = pd.DataFrame(vars, index=ts)

        # depricated:
        # self.data_frame = self.data_frame.append(df)
        # self.data_frame = pd.concat([self.data_frame, df])

        if "timestamp" in self.data_frame.columns:
            self.data_frame = self.data_frame.set_index("timestamp")

    def restore(self, persist_file, properties=None, converters=None):
        """
        restore data values from filesystem
        """

        if self.data_frame is None:
            self.data_frame = pd.DataFrame(columns=self.variables.keys())

        if persist_file is not None:
            self.data_frame = pd.read_csv(persist_file,
                                          parse_dates=True,
                                          infer_datetime_format=True,
                                          names=properties,
                                          converters=converters)

        if "timestamp" in self.data_frame.columns:
            self.data_frame = self.data_frame.set_index("timestamp")

    def persist(self, persist_file):

        if self.data_frame is None:
            raise ValueError("Resource.data_frame is None")

        if "timestamp" in self.data_frame.columns:
            self.data_frame = self.data_frame.set_index("timestamp")

        self.data_frame.to_csv(persist_file, mode='w')

    @property
    def dataframe(self):
        if self.data_frame is None:
            raise ValueError("no data in resource or no_snapshot is True")

        return self.data_frame

    async def poll(self):
        return True

    def to_json(self):
        vars = self.variables
        vars.update({"name": self.name})
        return json.dumps(vars)


def test_subscribe2():

    test_resource = Resource("foo", ["bar"])

    bar = test_resource.subscribe2("bar")

    assert bar() is None, "bar is not set yet. should be None"

    test_resource.setValue("bar", 100)

    assert bar() == 100, "bar should be 100.  is {}".format(bar())


def test_mqtt_auto_export():

    from aizero.mqtt_resource import MqttResource

    Resource.auto_export_mqtt(True, mqtt_export_options={
                              "retain_msgs": False})

    resource = Resource("MqttResource",
                        ["test1"])

    resource_sub = MqttResource("MqttResourceSubscriber", "localhost",
                                ["test1"],
                                variable_mqtt_map={"test1":
                                                   "MqttResource/test1"})

    loop = asyncio.get_event_loop()
    resource.setValue("test1", "failing")
    loop.run_until_complete(resource_sub.wait_until_connected())
    loop.run_until_complete(asyncio.sleep(5))

    assert resource_sub.getValue("test1") == "failing"


def test_mqtt_late_export():

    from aizero.mqtt_resource import MqttResource

    resource = Resource("MqttResource12312312",
                        ["test1"])

    resource_sub = MqttResource("MqttResourceSubscriber123123", "localhost",
                                ["test1"],
                                variable_mqtt_map={"test1":
                                                   "MqttResource12312312/test1"})

    loop = asyncio.get_event_loop()
    resource.setValue("test1", "failing")

    loop.run_until_complete(resource_sub.wait_until_connected())

    assert resource_sub.get_value("test1") != "failing"

    resource.export_mqtt()

    resource.setValue("test1", "winning")

    loop.run_until_complete(asyncio.sleep(5))

    assert resource_sub.get_value("test1") == "winning"


def test_persist_restore():
    persist_file = "./resource_test.csv"

    if os.path.isfile(persist_file):
        os.remove(persist_file)

    a1 = Resource("Resource", variables=["a", "b", "c"])

    a1.set_value("a", 1)
    a1.set_value("b", 2)
    a1.set_value("c", 3)

    a1.persist(persist_file)
    print(a1.dataframe)

    a2 = Resource("Resource2", variables=["a", "b", "c"])

    def conv(x):
        try:
            return int(x)
        except:
            pass

    a2.restore(persist_file, converters={"a": conv,
                                         "b": conv,
                                         "c": conv})
    print(a2.dataframe)

    assert len(a1.dataframe) == len(a2.dataframe)

    if os.path.isfile(persist_file):
        os.remove(persist_file)


def test_persist_file_not_found():
    a1 = Resource("Resource_test_persist_file_not_found",
                  variables=["a", "b", "c"])

    has_value_error = False

    try:
        a1.persist(None)
    except ValueError:
        has_value_error = True

    assert has_value_error


def test_snapshot():

    import uuid
    a1 = Resource(uuid.uuid4().hex, variables=["a", "b"])

    a1.set_value("a", 1)
    a1.set_value("a", 2)
    a1.set_value("a", 3)
    a1.set_value("a", 4)
    a1.set_value("b", [])

    print(a1.dataframe.head())
    print(a1.dataframe.to_numpy()[1])

    vals = a1.dataframe['a'].to_numpy()

    assert vals[0] == 1
    assert vals[1] == 2
    assert vals[2] == 3
    assert vals[3] == 4


def test_set_get_value():
    import uuid
    a1 = Resource(uuid.uuid4().hex, variables=['a'])

    a1.set_value("a", 1)

    assert a1.get_value('a') == 1


def test_properties():
    import uuid
    a1 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])

    assert 'a' in a1.properties
    assert 'b' in a1.properties
    assert 'z' in a1.properties


def test_dataframe_exception():
    import uuid
    a1 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])

    got_exception = False
    try:
        foo = a1.dataframe
    except ValueError:
        got_exception = True

    assert got_exception

    a1.set_value('a', 1)

    got_exception = False

    try:
        foo = a1.dataframe
    except ValueError:
        got_exception = True

    print(foo)

    # We set a value therefore, we should not get exception
    assert not got_exception


def test_bind_to():
    import uuid

    a1 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])
    a2 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])

    a1.bind_to("a", a2)

    a1.set_value("a", 123)

    assert a2.get_value("a") == 123

    a1.bind_to("a", a2, "z")

    assert a2.get_value("z") == 123

    a1.set_value("a", 456)

    assert a2.get_value("z") == 456


def test_bind_from():
    import uuid

    a1 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])
    a2 = Resource(uuid.uuid4().hex, variables=['a', 'b', 'z'])

    a1.bind_from("a", a2)

    a2.set_value("a", 123)

    assert a1.get_value("a") == 123

    a1.bind_from("a", a2, "z")

    assert a1.get_value("z") == 123

    a2.set_value("a", 456)

    assert a1.get_value("z") == 456


def main():
    test_subscribe2()
    test_mqtt_auto_export()
    test_persist_restore()
    test_persist_file_not_found()
    test_set_get_value()
    test_snapshot()


if __name__ == "__main__":
    main()
