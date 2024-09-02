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

async def detection_process(kachaka: KachakaFrame):
    # Activate camera and start timing
    st = time.time()
    await kachaka.human_detection()  # Detect human asynchronously

    # Process the face detection in parallel with other tasks
    face_process_task = asyncio.gather(kachaka.face_detector.process(kachaka.cv_img))
    navigate_task = asyncio.gather(asyncio.create_task(kachaka.short_navigate()))

    # cancel navigation task
    if kachaka.cd > 10:
        kachaka.cd = 0
        navigate_task.cancel()

    # Stalking stage
    if kachaka.target_found:
        # asyncio.gather(asyncio.create_task(kachaka.short_navigate()))
                
        await kachaka.adjust()

        if kachaka.face_detector.is_face_present():
            kachaka.face_found_count += 1
        else:
            kachaka.face_found_count -= 1

        if kachaka.face_found_count > 10:
            print("FACE FOUND")
            await kachaka.speak("こんにちは！")
            kachaka.target_found = False
            kachaka.face_found_count = 5
            kachaka.cd += 1
        elif kachaka.face_found_count <= 0:
            print("target lost")
            await kachaka.speak("顔が見つかりませんでした")
            kachaka.target_found = False
            kachaka.face_found_count = 5
            kachaka.cd += 1
    else:
        kachaka.face_found_count = 5
        
    # Ensure face processing and landmark drawing are done
    await face_process_task
    await navigate_task

    if kachaka.face_detector.is_face_present():
        kachaka.face_detector.draw_landmarks(kachaka.cv_img)

    # Annotation task
    await kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True)

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    # navigate_tasks = [asyncio.create_task(kachaka.navigate()) for kachaka in kachakas]
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    while any([kachaka.run for kachaka in kachakas]):
        await asyncio.gather(
            *[asyncio.create_task(detection_process(kachaka)) for i,kachaka in enumerate(kachakas)]
        )
        await display_kachakas(kachakas)

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    move_tasks = [asyncio.create_task(kachaka.move()) for kachaka in kachakas]
    monitor_task = asyncio.create_task(object_monitor_key_press(kachakas))
    controller_task = asyncio.create_task(controller(kachakas))
    print(f"{C.BLUE}Starting{C.RESET} Script")
    await asyncio.gather(
        controller_task,
        *move_tasks,
        monitor_task
        )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())