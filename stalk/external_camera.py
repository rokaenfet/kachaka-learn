import cv2
import asyncio
import warnings
import argparse
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    # 1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

CAMERA_INDEX = 1
SEARCH_PERIOD = 20 # iteration threshold for if face is found
SEARCH_COOLDOWN = 20 # cooldown between locking onto 2 instances of detected human
DURATION_FOR_CANCEL_NAV = 3 # duration until navigation is turned off when human is detected
WINDOW_NAME = "full"

async def detection_process(kachaka: KachakaFrame):
    st = time.time()
    await kachaka.human_detection()
    await asyncio.gather(
        kachaka.face_detector.process(kachaka.cv_img),
        kachaka.mp_landmark_model.process(kachaka.cv_img)
        )

    # Annotation task
    await asyncio.gather(
        kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True), # fps, lidar_dist, id
        # kachaka.human_detection_annotate(),
        # kachaka.mp_landmark_annotate(), # mebow annotation
        # kachaka.mp_landmark_model.draw_landmarks_on_image(kachaka.cv_img)
        )

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    # load external camera
    # camera_index = asyncio.new_event_loop().run_until_complete(get_camera_index()) ONLY RUNS ON NEWER CPU ON WINDOWS
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if cap.isOpened():
        while any([kachaka.run for kachaka in kachakas]):
            ret, image = cap.read()
            tasks = [asyncio.create_task(kachaka.move()) for kachaka in kachakas]
            if ret:
                for i,kachaka in enumerate(kachakas):
                    kachaka.cv_img = image.copy()
                    tasks.append(asyncio.create_task(detection_process(kachaka)))
            await asyncio.gather(
                *tasks
            )
            if ret:
                cv2.imshow(WINDOW_NAME, await display_kachakas(kachakas))
            cv2.waitKey(1)
    else:
        print(f"{C.RED}Failed{C.RESET} to access camera")
        for kachaka in kachakas:
            kachaka.run = False
    cap.release()
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