import cv2
import asyncio
import keyboard

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    # 1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

async def detection_process(kachaka:KachakaFrame):
    st = time.time()
    await kachaka.human_detection() # detect human
    kachaka.face_detector.process(kachaka.cv_img)

    # stalking stage
    if kachaka.target_found == True and kachaka.find_face_mode == False:
        prev_linear, prev_angular = kachaka.linear, kachaka.angular
        await kachaka.follow()
        if all([n==0 for n in [kachaka.linear, prev_linear, kachaka.angular, prev_angular]]):
            kachaka.human_found_count += 1
    else:
        kachaka.human_found_count = 0
        # kachaka.angular, kachaka.linear = 0.5, 0
    # identify if its been right next to the human for long enough
    if kachaka.human_found_count > 3:
        await kachaka.speak("見つけた！体を動かずに、顔をこっちに向けてください。")
        kachaka.find_face_mode = True
        kachaka.human_found_count = 0
    # looking for face
    if kachaka.find_face_mode == True:
        prev_linear, prev_angular = kachaka.linear, kachaka.angular
        await kachaka.adjust()
        if kachaka.face_detector.is_face_present():
            kachaka.face_found_count += 1
        else:
            kachaka.face_found_count = 0
    if kachaka.face_found_count > 5:
        print("FACE FOUND")
        await kachaka.speak("顔写真を撮りました！")
        kachaka.target_found = False
        kachaka.find_face_mode = False
    if kachaka.face_detector.is_face_present():
        kachaka.face_detector.draw_landmarks(kachaka.cv_img)
    await kachaka.emergency_stop()
    await kachaka.move()
    kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True) # annotate img

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    navigate_tasks = [asyncio.create_task(kachaka.navigate()) for kachaka in kachakas]
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    while True:
        detection_tasks = [asyncio.create_task(detection_process(kachaka)) for kachaka in kachakas]
        await asyncio.gather(
            *navigate_tasks,
            *detection_tasks
        )
        await display_kachakas(kachakas)
        asyncio.sleep(0.01)

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    controller_task = asyncio.create_task(controller(kachakas))
    monitor_task = asyncio.create_task(monitor_key_press(controller_task))
    print(f"{C.BLUE}Starting{C.RESET} Script")
    await asyncio.gather(
        controller_task,
        monitor_task
    )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())