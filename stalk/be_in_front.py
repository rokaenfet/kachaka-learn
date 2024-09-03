import cv2
import asyncio
import warnings
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    # 0:"192.168.118.158:26400",
    1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

SEARCH_PERIOD = 20 # iteration threshold for if face is found
SEARCH_COOLDOWN = 20 # cooldown between locking onto 2 instances of detected human
DURATION_FOR_CANCEL_NAV = 3 # duration until navigation is turned off when human is detected

async def detection_process(kachaka: KachakaFrame):
    st = time.time()
    await asyncio.gather(kachaka.human_detection())  # object detection
    await asyncio.gather(kachaka.face_detector.process(kachaka.cv_img)) # face detection
    await asyncio.gather(kachaka.mebow_model.process(kachaka.cv_img)) # HBOE / HOE

    # Annotation task
    await asyncio.gather(
        kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True), # fps, lidar_dist, id
        kachaka.mebow_annotate() # mebow annotation
        )

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    while any([kachaka.run for kachaka in kachakas]):
        await asyncio.gather(
            *[asyncio.create_task(detection_process(kachaka)) for i,kachaka in enumerate(kachakas)],
            *[asyncio.create_task(kachaka.move()) for kachaka in kachakas]
        )
        await display_kachakas(kachakas)

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