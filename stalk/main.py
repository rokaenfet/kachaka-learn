import cv2
import asyncio
import warnings
import time
from typing import List
from funcs import *  # Replace with specific imports to avoid wildcard imports

warnings.filterwarnings("ignore")

# Dictionary to store the IP addresses of the Kachaka robots
KACHAKA_IPS = {
    # 0: "192.168.118.158:26400",
    1: "192.168.118.159:26400",
    # 2: "192.168.118.77:26400"
}

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
    await asyncio.gather(kachaka.human_detection())  # Object detection
    await asyncio.gather(kachaka.face_detector.process(kachaka.cv_img))  # Face detection
    await asyncio.gather(kachaka.mp_landmark_model.process(kachaka.cv_img))  # Landmark detection (HBOE / HOE)
    
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
        kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True),  # FPS, LIDAR distance, ID
        kachaka.mp_landmark_annotate()  # MEBOW annotation
    )

async def controller(kachakas: List[KachakaFrame]) -> None:
    """
    Asynchronously runs the navigation and detection tasks for each Kachaka robot and displays
    the results. Navigation only runs if the Kachaka object has not found a target.
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    
    while any(kachaka.run for kachaka in kachakas):
        await asyncio.gather(
            *[asyncio.create_task(detection_process(kachaka)) for kachaka in kachakas]
        )
        await display_kachakas(kachakas)

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
