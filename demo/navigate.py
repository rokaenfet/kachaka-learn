import cv2
import asyncio
import keyboard
import sys
from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]

    while True:
        imgs = await get_map_images(kachakas)
        cv2.imshow("",np.concatenate(imgs, axis=1))
        for k in kachakas:
            pose = await k.async_client.get_robot_pose()
            x, y, theta = pose.x, pose.y, pose.theta
            # await k.async_client.move_to_pose()
        if keyboard.is_pressed("q"):
            break
        await asyncio.sleep(0.00001)  # prevent busy loop
        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())