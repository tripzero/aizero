from aizero.resource import Resource, MINS
from aizero.resource_py3 import Py3Resource as resource_poll

from hammock import Hammock

import sys
import traceback


class RemoteRestResource(Resource):
    def __init__(self, name, hammock_instance, poll_rate=2):
        self.hammock_instance = hammock_instance

        variables = self.hammock_instance.GET(verify=False).json()["variables"]

        Resource.__init__(self, name, variables.keys())
        self.poller = resource_poll(self.poll_func, MINS(poll_rate))

    def poll_func(self):
        try:
            variables = self.hammock_instance.GET(
                verify=False).json()["variables"]

            for key in variables.keys():
                self.setValue(key, variables[key])
        except Exception:
            print("error getting variables for RemoteRestResource {}".format(
                self.name))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=7, file=sys.stdout)


