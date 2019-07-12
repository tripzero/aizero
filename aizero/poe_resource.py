from .device_resource import DeviceResource
from .resource import MINS
import numpy as np
import asyncio


class FSTelnetSwitchProtocol:
    switches = None

    def __init__(self, address):
        if (FSTelnetSwitchProtocol.switches is not None and
                address in FSTelnetSwitchProtocol.switches):
            raise Exception("Singleton. don't init this class")

        from telnetlib import Telnet

        self.address = address

        if FSTelnetSwitchProtocol.switches is None:
            FSTelnetSwitchProtocol.switches = {}

        FSTelnetSwitchProtocol.switches[self.address] = self

        self.ready = False
        self.client = Telnet(address)

        try:
            self.login()
        except EOFError:
            print("failed to log into switch at {}".format(address))

    @staticmethod
    def instance(address):
        if (FSTelnetSwitchProtocol.switches is None or
                address not in FSTelnetSwitchProtocol.switches):
            return FSTelnetSwitchProtocol(address)

        else:
            return FSTelnetSwitchProtocol.switches[address]

    def write_str(self, data):
        self.client.write(data.encode('ascii') + b'\n')

    def read_until(self, data, timeout=None):
        self.client.read_until(data.encode('ascii'), timeout)

    def reconnect(self):
        self.client.close()
        self.client.open(self.address)
        self.login(self.username, self.password)

    def login(self, user="admin", password="admin"):
        self.username = user
        self.password = password

        self.client.read_until(b"Username:")
        self.write_str(user)
        self.client.read_until(b'Password:')
        self.write_str(password)

        self.client.read_until(b'Switch>')
        self.write_str('enter')  # enter 'Exec mode'
        self.client.read_until(b'Switch#')
        self.write_str('config')
        self.client.read_until(b'Switch_config#')

        self.ready = True

    def disable(self, port):
        self.enable(port, False)

    def enable(self, port, state=True):
        if not self.ready:
            return

        iface_str = "g0/{}".format(port)

        self.write_str("interface {}".format(iface_str))

        self.read_until('Switch_config_g0/{}#'.format(port), 1)

        disable = ""

        if state:
            disable = "no "

        self.write_str("{}poe disable".format(disable))

        self.read_until('Switch_config_g0/{}#'.format(port), 1)

        self.write_str("exit")

        self.read_until('Switch_config#'.format(port), 1)

    def power_usage(self, port=None):
        if not self.ready:
            return None

        num_ports = 8

        try:
            self.write_str('show poe power')
        except BrokenPipeError:
            # probably lost connection. reconnect
            self.reconnect()
            return

        try:

            data = self.client.read_until(b'Switch_config#', 1)
        except EOFError:
            self.reconnect()
            return

        try:
            data = data.replace(b'show poe power', b'')
            data = data.replace(b'\r\n', b' ')
            data = data.replace(b' ', b'\t')
            data = data.replace(b'mW', b'')
            data = data.replace(b'Port', b'')
            data = data.replace(b'Current', b'')
            data = data.replace(b'Max', b'')
            data = data.replace(b'Average', b'')
            data = data.replace(b'Peak', b'')
            data = data.replace(b'Bottom', b'')
            data = data.replace(b'Switch_config#', b'')
            data = data.split(b'\t')
            data = [a for a in data if a != b'']

            data = np.array(data)
            data = np.reshape(data, (num_ports, 6))

            if port is None:
                power = np.sum(np.array(data[:, 1], dtype="int")) / 100.0

            else:
                power = int(data[port - 1, 1]) / 100.0

            return power

        except IndexError as ie:
            print(ie)
            print("probably port too high for switch")
            print(data)
        except ValueError as ve:
            print(ve)
            print(data)


class PoeResource(DeviceResource):

    def __init__(self, address, port, **kwargs):
        super().__init__(**kwargs)

        self.switch = FSTelnetSwitchProtocol.instance(address)
        self.port = port

    def run(self):

        if not super().run():
            return False

        self.switch.enable(self.port)

    def stop(self):

        if not super().stop():
            return False

        self.switch.disable(self.port)

    @asyncio.coroutine
    def poll(self):

        while True:
            power = self.switch.power_usage(self.port)
            if power is not None:
                self.update_power_usage(power)
            else:
                power = 0
                self.update_power_usage(0)

            if power > 0:
                super().run()
            else:
                super().stop()

            yield from asyncio.sleep(MINS(3))


def get_switch_devices(address, num_ports=8):
    devs = []
    for i in range(num_ports):
        dev = PoeResource(
            address, i, name="poe_port_{}_{}".format(i, address))
        devs.append(dev)

    return devs


def test_power():
    switch = FSTelnetSwitchProtocol.instance("192.168.1.2")

    print("power usage: {}".format(switch.power_usage()))
    print("power usage port 4: {}".format(switch.power_usage(port=4)))


def test_enable():
    import time
    switch = FSTelnetSwitchProtocol.instance("192.168.1.2")

    switch.enable(4)

    time.sleep(5)

    print("power usage 4: {}mW".format(switch.power_usage(4)))

    switch.disable(4)

    time.sleep(5)

    assert switch.power_usage(4) == 0

    switch.enable(4)


if __name__ == "__main__":
    test_power()
    test_enable()
