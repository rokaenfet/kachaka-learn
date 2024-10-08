import time
import cv2
import kachaka_api
import numpy as np
import asyncio
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output, display
from pynput import keyboard
import sys

KACHAKA_IP = "192.168.118.159:26400"

speed = .3
turn = .5

keys = {
    "w":(speed,0),
    "s":(-speed,0),
    "a":(0,turn),
    "d":(0,-turn)
}

pressed = {
    "w":False,
    "s":False,
    "a":False,
    "d":False
}


def move(client, linear, angular):
    client.set_robot_velocity(linear=linear, angular=angular)

async def camera(client: kachaka_api.aio.KachakaApiClient):
    async for image in client.front_camera_ros_compressed_image.stream():
        st = time.time()
        cv_image = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        cv2.putText(cv_image, f"fps:{round(1/(time.time()-st))}", (20,80), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        cv2.imshow("", cv2.resize(cv_image, (800,600)))
        cv2.waitKey(1)

async def get_and_show_laser_scan_loop(client: kachaka_api.aio.KachakaApiClient):
    async for scan in client.ros_laser_scan.stream():
        fig = plt.figure(figsize=(5, 5))

        n = len(scan.ranges)
        x = list(range(n))
        y = list(range(n))
        for i in range(n):
            theta = scan.angle_min + scan.angle_increment * i
            x[i] = -scan.ranges[i] * math.cos(theta)
            y[i] = -scan.ranges[i] * math.sin(theta)

        plt.plot(0, 0, "o", color="black")
        plt.plot(x, y, ".", markersize=5)
        plt.xlim(-6.0, 6.0)
        plt.ylim(-6.0, 6.0)
        plt.grid(True)
        plt.gca().set_aspect("equal", adjustable="box")
        fig.savefig("demo/live_lidar.png")

async def main():
    async_client = kachaka_api.aio.KachakaApiClient(KACHAKA_IP)
    sync_client = kachaka_api.KachakaApiClient(KACHAKA_IP)
    lidar_task = asyncio.create_task(get_and_show_laser_scan_loop(async_client))
    camera_task = asyncio.create_task(camera(async_client))

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ).start()
    try:
        while True:
            for k,v in pressed.items():
                if v == True:
                    print(k,v,keys[k])
                    await async_client.set_robot_velocity(*keys[k])
                    # move(sync_client, *keys[k])
            
            status_msg = ''.join([k for k,v in pressed.items() if v==True])
            sys.stdout.write("\r"+status_msg)
            sys.stdout.flush()

            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        pass

    # lidar_task.cancel()
    # camera_task.cancel()

def on_press(key):
    try:
        key = key.char
    except AttributeError:
        return
    if key in keys:
        pressed[key] = True

def on_release(key):
    try:
        key = key.char
    except AttributeError:
        return
    if key in keys:
        pressed[key] = False

asyncio.run(main())