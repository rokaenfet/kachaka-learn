import cv2
import asyncio
import warnings
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    # 1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

async def nav(kachaka:KachakaFrame):
    while True:
        if not await kachaka.check_navigate():
            print("nav")
            await kachaka.short_navigate()

async def watch_nav(kachaka:KachakaFrame):
    while True:
        print(await kachaka.check_navigate())

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    navigate_tasks = [asyncio.create_task(nav(kachaka)) for kachaka in kachakas]
    watch_navigate_tasks = [asyncio.create_task(watch_nav(kachaka)) for kachaka in kachakas]
    tasks = navigate_tasks + watch_navigate_tasks
    monitor_task = task_monitor_key_press(tasks)

    await asyncio.gather(*tasks, monitor_task)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())