
from aizero.resource import Resource

import json
import sys
import traceback


class ConfigurationResource(Resource):

    def __init__(self, configFileName):
        Resource.__init__(self, "ConfigurationResource", ["config"])
        self.configFile = configFileName

        self.reload()

        self.setValue("config", self.config)

        # self.subscribe("config", lambda v: self.writeConfig())

        # Resource.makeCallable(self.name, "setValue", self.setValue)

    def reload(self):
        import json

        with open(self.configFile) as f:
            self.config = json.loads(f.read())

    def writeConfig(self):
        with open(self.configFile, "w") as f:
            try:
                f.write(json.dumps(self.config,
                                   sort_keys=True,
                                   indent=4,
                                   separators=(',', ': ')))
            except Exception:
                print("Failed to write config: {0}".format(self.config))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=6, file=sys.stdout)

    def set_value(self, key, value):
        if key == "config":
            Resource.set_value(self, "config", self.config)
            return

        self.config[key] = value
        self.writeConfig()

    def get_value(self, key, defaultValue=None):
        if key not in self.config and defaultValue:
            # "create it if it doesn't exist and there's a specified default"
            self.set_value(key, defaultValue)

        return self.config[key]
