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

LOCATIONS = ["start","end"]
target_loc_i = [0 for _ in range(len(KACHAKA_IPS))]

async def show_map(kachakas):
    while True:
        imgs = await get_map_images(kachakas)
        cv2.imshow("",np.concatenate(imgs, axis=1))
        cv2.waitKey(1)

async def navigate(kachaka:KachakaFrame):
    locations = kachaka.get_locations(LOCATIONS)
    i = 0
    while True:
        result = await kachaka.async_client.move_to_location(locations[i].id)
        if result.success:
            i = (i+1)%len(locations)
        else:
            print(kachaka.error_code[result.error_code])

async def monitor_key_press(tasks:list[asyncio.Task]):
    while True:
        key = await aioconsole.ainput()
        if key.lower() == 'q':
            print("Key 'q' pressed. Terminating all tasks...")
            for task in tasks:
                task.cancel()
            break

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    tasks = [asyncio.create_task(navigate(k)) for k in kachakas] + [asyncio.create_task(show_map(kachakas))]
    monitor_task = asyncio.create_task(monitor_key_press(tasks))

    await asyncio.gather(*tasks, monitor_task)

if __name__ == "__main__":
    asyncio.run(main())