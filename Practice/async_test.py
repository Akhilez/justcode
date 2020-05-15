import asyncio


async def count():
    print(1)
    await asyncio.sleep(1)
    print(2)


async def main():
    # for _ in range(3):
    await asyncio.gather(count(), count(), count())


asyncio.run(main())
