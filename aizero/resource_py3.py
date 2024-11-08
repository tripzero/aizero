import asyncio


class Py3Resource:

    def __init__(self, poll_func, sleep_timer, is_coroutine=False):
        self.poll_func = poll_func
        self.sleep_timer = sleep_timer
        self.is_coroutine = is_coroutine

        asyncio.get_event_loop().create_task(self.poll())

    async def poll(self):
        while True:
            if self.is_coroutine:
                await self.poll_func()
            else:
                futures = self.poll_func()

                if futures is not None:
                    for future in futures:
                        await future

            await asyncio.sleep(self.sleep_timer)


class Py3PollWhileTrue:

    def __init__(self, poll_func, sleep_timer):
        self.poll_func = poll_func
        self.sleep_timer = sleep_timer

        asyncio.get_event_loop().create_task(self.poll())

    async def poll(self):
        while self.poll_func():
            await asyncio.sleep(self.sleep_timer)
