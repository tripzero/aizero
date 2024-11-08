import asyncio
from hammock import Hammock as SolarEdge
import json

import sys
import traceback

from .resource import Resource, MINS, get_resource
from .resource_py3 import Py3Resource as resource_poll
from .utils import run_thread


class SolarPower(Resource):

    def __init__(self):
        Resource.__init__(self, "SolarPower", [
                          "current_power",
                          "available_power",
                          "energy_day",
                          "energy_month",
                          "energy_lifetime",
                          "energy_year"])

        self.solar_edge = SolarEdge('https://monitoringapi.solaredge.com')

        config_rsrc = get_resource("ConfigurationResource")

        self.api_key = config_rsrc.get_value("solaredge_api")
        self.format = 'application/json'

        self.id = self._get_id()

        self.poller = resource_poll(self.poll_func, MINS(5), is_coroutine=True)

    def _get_id(self):
        resp = self.solar_edge.sites.list.GET(
            params={'api_key': self.api_key, 'format': self.format})
        if resp.status_code != 200:
            return None

        sites = json.loads(resp.content)

        return sites['sites']['site'][0]['id']

    def _get_overview(self):
        params = {'api_key': self.api_key, 'format': self.format}

        resp = self.solar_edge.site(self.id).overview.GET(params=params)

        print("solar status code: {}".format(resp.status_code))
        print("solar content: {}".format(resp.content))

        if resp.status_code != 200:
            return None, None, None, None

        overview = json.loads(resp.content)

        energy_lifetime = overview['overview']['lifeTimeData']['energy']
        energy_year = overview['overview']['lastYearData']['energy']
        energy_month = overview['overview']['lastMonthData']['energy']
        energy_day = overview['overview']['lastDayData']['energy']
        current_power = overview['overview']['currentPower']['power']

        return (energy_lifetime,
                energy_year,
                energy_month,
                energy_day,
                current_power)

    async def poll_func(self):
        try:
            energy_lifetime, energy_year, energy_month, energy_day, current_power = await run_thread(
                self._get_overview)
            self.set_value("energy_lifetime", energy_lifetime)
            self.set_value("energy_year", energy_year)
            self.set_value("energy_month", energy_month)
            self.set_value("energy_day", energy_day)
            self.set_value("current_power", current_power)
            self.set_value("available_power", current_power)
        except Exception:
            print("solar: error getting overview")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=6, file=sys.stdout)


if __name__ == "__main__":
    solar = SolarPower()

    print('id = {}'.format(solar.id))

    print('power = {}'.format(solar._get_overview()))
