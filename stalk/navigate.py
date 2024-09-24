import cv2
import asyncio
import warnings
import argparse
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.168:26400",
    1:"192.168.118.166:26400",
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
        await asyncio.gather(*[kachaka.get_image_from_camera() for kachaka in kachakas])
        if all([kachaka.cv_img is not None for kachaka in kachakas]):
            image = await get_display_image(kachakas)
            cv2.imshow(WINDOW_NAME, image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

async def get_display_image(kachakas:list[KachakaFrame]):
    a = []
    for kachaka in kachakas:
        padded = pad_images_to_same_shape([kachaka.cv_img, await kachaka.draw_map()])
        a.append(np.concatenate(padded, axis=1))
    image = np.concatenate(a, axis=0) if len(a) > 0 else a[0]
    image = image_resize(image, width=SCREEN_W, height=SCREEN_H)
    return image

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    # navigate_tasks = [asyncio.create_task(kachaka.short_navigate()) for kachaka in kachakas]
    monitor_task = asyncio.create_task(object_monitor_key_press(kachakas))
    controller_task = asyncio.create_task(controller(kachakas))

    print(f"{C.BLUE}Starting{C.RESET} Script")
    await asyncio.gather(
        controller_task,
        monitor_task,
        )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())