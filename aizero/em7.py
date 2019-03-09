import numpy as np
from datetime import datetime


def remove_nan(data):
    return filter(lambda v: v == v, data)


def totimestamp(dt, epoch=datetime(1970, 1, 1)):
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6


def get_data(device_name, app_name, property_name, credentials,
             duration=None, begin_timestamp=None, end_timestamp=None,
             verbose=False, normalized=None):

    from sciencelogic.client import Client

    if isinstance(begin_timestamp, datetime):
        begin_timestamp = totimestamp(begin_timestamp)

    if isinstance(end_timestamp, datetime):
        end_timestamp = totimestamp(end_timestamp)

    c = Client(*credentials)

    device = None

    if not isinstance(device_name, int):
        devices = c.devices(details=True)

        for dev in devices:
            if dev.description == device_name:
                device = dev
                break

        if not device:
            print("could not find device {}".format(device_name))
            print("available devices: {}".format(devices))
            return None
    else:
        try:
            device = c.get_device(device_name)
        except Exception:
            print("could not find device {}".format(device_name))
            return None

    counter = None
    counter_names = []

    for dev_counter in device.performance_counters():
        cn = dev_counter.name()

        if cn[0] == ' ':
            cn = cn.replace(' ', '')

        counter_names.append(cn)
        if cn == app_name:
            counter = dev_counter
            break

    if not counter:
        print("could not find app for {}".format(app_name))
        print("available apps: {}".format(counter_names))
        return None

    data_map = None

    presentations = counter.get_presentations()
    presentation_names = []

    for presentation in presentations:
        presentation_names.append(presentation.name)

        if verbose:
            print("presentation.name='{}' == property_name='{}'? ".format(
                presentation.name, property_name))

        if str(presentation.name) == str(property_name):
            if verbose:
                print("found matching presentation name!")
            if normalized:
                presentation.data_uri = presentation.data_uri.replace(
                    "data?duration=24h", "normalized_hourly?duration=24h")

            data_map = presentation.get_data(
                beginstamp=begin_timestamp,
                endstamp=end_timestamp,
                duration=duration)

            if normalized and 'avg' in data_map:
                data_map = data_map['avg']

            break

    if data_map is None:
        print("could not find presentation '{}'' or no data for that period".format(
            property_name))
        if property_name not in presentation_names:
            print("available presentation names: '{}'".format(
                "', '".join(presentation_names)))
        return None

    if verbose:
        print(data_map.keys())

    data_map_size = 0
    data_map_flatter = {}

    # combine all the index keys:

    for index_key in list(data_map.keys()):
        for timestamp, value in data_map[index_key].items():
            data_map_flatter[timestamp] = value

    data_map = data_map_flatter

    data = np.zeros((len(list(data_map)), 2), np.float32)

    data[:, 0] = np.array(list(data_map)).astype(float)
    data[:, 1] = np.array(list(data_map.values())).astype(float)

    return data


if __name__ == "__main__":

    dev = "redStrip"
    counter = "AirTemperature"
    property_name = "airTemp"

    print(get_data(dev, counter, property_name))
