import cv2
import asyncio
import warnings
import time
from typing import List
from funcs import *  # Replace with specific imports to avoid wildcard imports

warnings.filterwarnings("ignore")

# Dictionary to store the IP addresses of the Kachaka robots
KACHAKA_IPS = {
    0:"192.168.118.168:26400",
    1:"192.168.118.166:26400",
    # 2:"192.168.118.160:26400"
    }
WINDOW_NAME = "f"

# Constants for detection and navigation logic
SEARCH_PERIOD = 20  # Iteration threshold for detecting if a face is found
SEARCH_COOLDOWN = 20  # Cooldown between locking onto two instances of a detected human
DURATION_FOR_CANCEL_NAV = 3  # Duration until navigation is turned off when a human is detected

async def detection_process(kachaka: KachakaFrame) -> None:
    """
    Handles the detection process for a single Kachaka robot, including object, face,
    and landmark detection. If a human is detected, the robot cancels navigation and
    enters a 'stalking' mode where it adjusts its position to follow the target.
    """
    st = time.time()
    
    # Run the detection processes asynchronously
    await kachaka.human_detection()
    await asyncio.gather(
        kachaka.face_detector.process(kachaka.cv_img),
        kachaka.mp_landmark_model.process(kachaka.cv_img)
        )
    
    # Stalking stage: React to detected target
    if kachaka.target_found:
        # Cancel navigation if necessary and prevent future navigations
        if kachaka.cd > DURATION_FOR_CANCEL_NAV:
            if await kachaka.check_navigate():
                await kachaka.async_client.cancel_command()
            kachaka.run_nav = False
        
        # Print debug information
        print(
            f"ID:{kachaka.id} | .cd:{kachaka.cd} | .face_found_count:{kachaka.face_found_count} "
            f"| .run_nav:{kachaka.run_nav} | (.linear,.angular):"
            f"({round(kachaka.linear, 3)}, {round(kachaka.angular, 3)})"
        )
        
        kachaka.cd += 1
        if kachaka.cd > SEARCH_COOLDOWN:
            await kachaka.adjust()
            await kachaka.move()
            
            if kachaka.face_detector.is_face_present():
                kachaka.face_detector.draw_landmarks(kachaka.cv_img)
                kachaka.face_found_count += 1
            else:
                kachaka.face_found_count -= 1

            if not (0 < kachaka.face_found_count <= SEARCH_PERIOD):
                kachaka.target_found = False
                kachaka.face_found_count = SEARCH_PERIOD // 2
                kachaka.cd = 0
                
                if kachaka.face_found_count > SEARCH_PERIOD:
                    print("Human found")
                    await kachaka.speak("こんにちは！")
                elif kachaka.face_found_count <= 0:
                    print("Target lost")
                    await kachaka.speak("顔が見つかりませんでした")
    else:
        kachaka.face_found_count = SEARCH_PERIOD // 2
        kachaka.run_nav = True
        kachaka.cd = 0

    # Annotation task
    await asyncio.gather(
        kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True), # fps, lidar_dist, id
        kachaka.human_detection_annotate(), # YOLO
        kachaka.mp_landmark_annotate(), # MP Landmark HOE
        kachaka.mp_landmark_model.draw_landmarks_on_image(kachaka.cv_img) # draw landmarks
        )

async def controller(kachakas: List[KachakaFrame]) -> None:
    """
    Asynchronously runs the navigation and detection tasks for each Kachaka robot and displays
    the results. Navigation only runs if the Kachaka object has not found a target.
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")

    # load external camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # doesnt work
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # doesnt work
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if cap.isOpened():
        while any([kachaka.run for kachaka in kachakas]):
            ret, image = cap.read()
            tasks = [asyncio.create_task(kachaka.move()) for kachaka in kachakas]
            for i,kachaka in enumerate(kachakas):
                kachaka.cv_img = image.copy()
                tasks.append(asyncio.create_task(detection_process(kachaka)))
            await asyncio.gather(
                *tasks
            )
            cv2.imshow(WINDOW_NAME, await display_kachakas(kachakas))
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)
            cv2.waitKey(1)
    else:
        print(f"{C.RED}Failed{C.RESET} to access camera")
        for kachaka in kachakas:
            kachaka.run = False
    cap.release()
    cv2.destroyAllWindows()

async def navigator(kachaka: KachakaFrame) -> None:
    """
    Continuously checks and manages the navigation process for a single Kachaka robot.
    Cancels navigation if needed or triggers short navigations based on conditions.
    """
    while kachaka.run:
        nav_running = await kachaka.check_navigate()
        
        if kachaka.run_nav:
            if not nav_running:
                await kachaka.short_navigate()
        else:
            await kachaka.cancel_navigation()

async def main() -> None:
    """
    The main function initializes Kachaka clients and manages the overall process
    of navigation, monitoring, and controlling the robots.
    """
    # Initiate clients for each Kachaka IP
    kachakas = [KachakaFrame(v, k) for k, v in KACHAKA_IPS.items()]
    
    # Create asyncio tasks for navigation and monitoring
    navigate_tasks = [asyncio.create_task(navigator(kachaka)) for kachaka in kachakas]
    monitor_task = asyncio.create_task(object_monitor_key_press(kachakas))
    controller_task = asyncio.create_task(controller(kachakas))

    print(f"{C.BLUE}Starting{C.RESET} Script")
    
    # Run the tasks concurrently
    await asyncio.gather(
        controller_task,
        *navigate_tasks,
        monitor_task
    )

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
