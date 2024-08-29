import time
import cv2
import kachaka_api
import numpy as np
import asyncio
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import keyboard
import sys
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image

from funcs import *

KACHAKA_IP = "192.168.118.158:26400"
LINEAR_SPEED = 0.3
AUTO_LINEAR_SPEED = 0.2
ANGULAR_SPEED = 1
AUTO_ANGULAR_SPEED = 0.1
EMERGENCY_STOP_DISTANCE = 0.2
WIN_W = 1280
WIN_H = 720
THRE = 30

async def main():
    # initiate clients
    async_client = kachaka_api.aio.KachakaApiClient(KACHAKA_IP)
    sync_client = kachaka_api.KachakaApiClient(KACHAKA_IP)

    # LOAD
    stream_i = async_client.front_camera_ros_compressed_image.stream()
    stream_d = async_client.object_detection.stream()
    undistort_map = get_camera_info(sync_client)

    # VARS
    emergency_stop = False
    target_found = False

    # no freedom for kachaka...
    await async_client.set_manual_control_enabled(True)

    # async tasks if any

    # main loop
    while True:
        st = time.time()
        await async_client.set_robot_velocity(0, 0) # kachaka maintains vel last given for 0.3s if no further instructions given
        linear, angular = 0, 0
        if emergency_stop == False:
            # manual control
            if keyboard.is_pressed("w"): linear += LINEAR_SPEED
            if keyboard.is_pressed("s"): linear -= LINEAR_SPEED
            if keyboard.is_pressed("a"): angular += ANGULAR_SPEED
            if keyboard.is_pressed("d"): angular -= ANGULAR_SPEED
        if target_found == True and keyboard.is_pressed("space") == True:
            tx, ty, tw, th = target_pos
            center_tx, center_ty = tx+tw//2, ty+th//2
            # horizontal adjustment
            angular = np.sign(WIN_W//2-center_tx)*AUTO_ANGULAR_SPEED
            # distance adjustment
            if ty <= THRE:
                linear -= AUTO_LINEAR_SPEED
            elif th <= WIN_H//2.5:
                linear += AUTO_LINEAR_SPEED

        await async_client.set_robot_velocity(linear, angular)

        # end loop
        if keyboard.is_pressed("q"):
            break

        # emergency stop system
        lidar_scan = await async_client.get_ros_laser_scan()
        nearest_scan_dist = min([dist for dist in lidar_scan.ranges if dist > 0])
        if nearest_scan_dist < EMERGENCY_STOP_DISTANCE:
            emergency_stop = True
            sync_client.speak("離れてください")
            sync_client.set_robot_velocity(0, 0)
        elif emergency_stop == True:
            emergency_stop = False
            sync_client.speak("再開します")

        # human detection
        image, (header, objects) = await asyncio.gather(anext(stream_i), anext(stream_d))
        cv_img = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1) # roscompressed to img
        cv_img = undistort(cv_img, *undistort_map) # undistort
        cv_img = draw_box(cv_img, objects) # draw bounding box
        if len(objects) > 0:
            target_pos = process_object(objects) #x, y, w, h
            if target_pos:
                cv2.putText(cv_img, "X", (target_pos[0]+target_pos[2]//2, target_pos[1]+target_pos[3]//2), *lazy_cv2_txt_params)
                target_found = True
            else:
                target_found = False
        else:
            target_found = False
        
        cv2.putText(cv_img, f"fps:{round(1/(time.time()-st))}", (20, 80), *lazy_cv2_txt_params)
        cv2.putText(cv_img, f"{round(nearest_scan_dist, 3)}", (20, 140), *lazy_cv2_txt_params)

        # display
        cv2.imshow("", cv_img)
        cv2.waitKey(1)

        # otherwise keyboard breaks
        await asyncio.sleep(0.00001)

    # end async task

if __name__ == "__main__":
    asyncio.run(main())