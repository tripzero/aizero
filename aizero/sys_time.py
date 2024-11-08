
import asyncio
import dateutil.parser
import pytz
import subprocess

from datetime import datetime, tzinfo

from aizero.resource import Resource, ResourceNotFoundException
from aizero.mqtt_resource import MqttResource, mqtt_topic_converter


async def publish_time():
    sys_time_resource = Resource("Date", ["datetime", "datetime_utc"])

    while True:
        time_str = subprocess.check_output(
            ["date", "-I", "seconds"], shell=True)

        time_str = time_str.decode('utf-8').replace("\n", "")

        sys_time_resource.setValue("datetime", time_str)

        time_str = subprocess.check_output(
            ["date", "-I", "seconds", "-u"], shell=True)

        time_str = time_str.decode('utf-8').replace("\n", "")

        sys_time_resource.setValue("datetime_utc", time_str)

        await asyncio.sleep(60)


@mqtt_topic_converter("Date/datetime")
def datetime_convert(value):
    return dateutil.parser.parse(value)


class CurrentTimeResource(MqttResource):

    def __init__(self, broker="localhost"):
        super().__init__("CurrentTimeResource", broker, ["datetime"],
                         variable_mqtt_map={
            "datetime": "Date/datetime"
        })


def get_current_datetime(utc=False):
    try:
        cur_time_resource = Resource.resource("CurrentTimeResource")

        if utc:
            u = cur_time_resource.get_value("datetime_utc").replace(
                tzinfo=pytz.utc)
            return u

        return cur_time_resource.get_value("datetime")

    except ResourceNotFoundException:
        if utc:
            u = datetime.utcnow().replace(tzinfo=pytz.utc)
            return u

        return datetime.now()


def test_isodate():
    parsed_date = dateutil.parser.parse(
        "2018-12-30T20:25:17")

    print(parsed_date)

    assert parsed_date == datetime(2018, 12, 30, 20, 25, 17)


def main():

    Resource.auto_export_mqtt(True)

    asyncio.get_event_loop().run_until_complete(publish_time())


if __name__ == "__main__":
    main()
