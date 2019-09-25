
import asyncio
from functools import partial


def run_thread(func, *args, **kwargs):

    loop = asyncio.get_event_loop()

    return loop.run_in_executor(None, partial(func, *args, **kwargs))
