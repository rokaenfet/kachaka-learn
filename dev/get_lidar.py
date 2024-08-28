import math

import kachaka_api
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output
import cv2, asyncio
KACHAKA_IP = "192.168.118.158:26400"

async def get_and_show_laser_scan_loop(client: kachaka_api.aio.KachakaApiClient):
    async for scan in client.ros_laser_scan.stream():
        clear_output(wait=True)
        fig = plt.figure(figsize=(5, 5))

        n = len(scan.ranges)
        x = list(range(n))
        y = list(range(n))
        for i in range(n):
            theta = scan.angle_min + scan.angle_increment * i
            x[i] = -scan.ranges[i] * math.cos(theta)
            y[i] = scan.ranges[i] * math.sin(theta)

        plt.plot(0, 0, "o", color="black")
        plt.plot(x, y, ".")
        plt.xlim(-6.0, 6.0)
        plt.ylim(-6.0, 6.0)
        plt.grid(True)
        plt.gca().set_aspect("equal", adjustable="box")
        fig.savefig("dev/live_lidar.png")

async def main():
    client = kachaka_api.aio.KachakaApiClient(KACHAKA_IP)
    await client.set_manual_control_enabled(True)
    await get_and_show_laser_scan_loop(client)

asyncio.run(main())