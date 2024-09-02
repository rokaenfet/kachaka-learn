import cv2
import asyncio
import keyboard
import sys
import aioconsole
from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    # 1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

target_loc_i = [0 for _ in range(len(KACHAKA_IPS))]

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    tasks = [asyncio.create_task(navigate(k)) for k in kachakas] + [asyncio.create_task(show_map(kachakas))]
    monitor_task = asyncio.create_task(monitor_key_press(tasks))

    await asyncio.gather(*tasks, monitor_task)

if __name__ == "__main__":
    asyncio.run(main())