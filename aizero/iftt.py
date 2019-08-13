import urllib.request
import json


def triggerEvent(eventName, key, val=None, val2=None, val3=None):
    url = "https://maker.ifttt.com/trigger/{}/with/key/{}"

    url = url.format(eventName, key)

    data = {"value1": val, "value2": val2, "value3": val3}

    data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data)
    resp = urllib.request.urlopen(req)

    return resp.read()


def test_trigger():

    resp = triggerEvent("Test", "fill_in_with_key")

    assert resp is not None
