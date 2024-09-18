import cv2
import asyncio
import warnings
import argparse
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.168:26400",
    # 1:"192.168.118.166:26400",
    # 2:"192.168.118.160:26400"
    }

CAMERA_INDEX = 1
SEARCH_PERIOD = 20 # iteration threshold for if face is found
SEARCH_COOLDOWN = 20 # cooldown between locking onto 2 instances of detected human
DURATION_FOR_CANCEL_NAV = 3 # duration until navigation is turned off when human is detected
WINDOW_NAME = "full"

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    # main loop
    while any([kachaka.run for kachaka in kachakas]):
        mover_tasks = [asyncio.create_task(kachaka.move()) for kachaka in kachakas]
        # display
        for kachaka in kachakas:
            depth_image, color_image = kachaka.realsense.get_frames()
            if depth_image is not None and color_image is not None:
                kachaka.cv_img = color_image.copy()
                kachaka.color_image = color_image
                kachaka.depth_image = depth_image
            else:
                kachaka.cv_img = kachaka.color_image = kachaka.depth_image = None
        await asyncio.gather(
            *[kachaka.test_circle() for kachaka in kachakas]
        )
        if all([kachaka.cv_img is not None for kachaka in kachakas]):
            cv2.imshow(WINDOW_NAME, await display_kachakas(kachakas))
        cv2.waitKey(1)
    cv2.destroyAllWindows()

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    monitor_task = asyncio.create_task(object_monitor_key_press(kachakas))
    controller_task = asyncio.create_task(controller(kachakas))

    print(f"{C.BLUE}Starting{C.RESET} Script")
    await asyncio.gather(
        controller_task,
        monitor_task
        )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())