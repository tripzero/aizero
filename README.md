# aizero
Framework for automation with machine learning capabilities

## What makes aizero unique?

Most automation systems tend to center around arbitrary automation. aizero
differs from the norm in that it has the capability to automate around power
saving. While this is sometimes possible to configure other automation systems
to do this, it can be a painful process.

Other supported features include
- device priority system
- device time of use modes
- device policies (for more complex automation)
- combine data from resources to create a machine learning model 
  (see examples/temperature_predictor.py)
- mqtt integration
- zwave support (incomplete)

## Get started

### Install:
Clone and:

```bash
sudo pip3 install .
```

### Write some code:

```python
from aizero.device_resource import DeviceManager, DeviceResource

device_manager = DeviceManager(max_power_budget=1000)

device1 = DeviceResource("my special device", power_usage=500)
device2 = DeviceResource("my other device", power_usage=500)

asyncio.get_event_loop().run_forever()
```

### Set device priority

```python
from aizero.device_resource import RuntimePriority

# specifying a priority makes this device 'managed'
device3 = DeviceResource("low priority device", power_usage=100, 
                         priority=RuntimePriority.low)

# run all devices
device1.run()
device2.run()
device3.run()

# devices 1 and 2 will run because they are not managed
print("device3 is running? {}".format(device3.running()))
print("device3 is running? {}".format(device3.running()))

# device3 will not be able to run because system will be over budget by 100W
print("device3 is running? {}".format(device3.running()))
```

### Add automation using a policy

```python
from aizero.device_resource import RunIfCanPolicy

device4 = DeviceResource("automated device",
                         power_usage=100,
                         runtime_policy=RunIfCanPolicy())

# At first, device4 will not run because our system is running at our budget
# of 1000W.
print("device4 running? {}".format(device4.running()))

# Lets stop one of our devices and watch the automatic automation:
device1.stop()

# device4 should start running automatically
print("device4 running? {}".format(device4.running()))
```

## Supported third-party systems

- ecobee thermostats and sensors
- kasa switches and plugs
- zwave locks
- solar edge inverters
- open weathermap
- fs.net POE network switches
- myq garage door openers

