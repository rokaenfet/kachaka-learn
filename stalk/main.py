import cv2
import asyncio
import warnings
warnings.filterwarnings("ignore")

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

SEARCH_PERIOD = 40
SEARCH_COOLDOWN = 20
DURATION_FOR_CANCEL_NAV = 3

async def detection_process(kachaka: KachakaFrame):
    st = time.time()
    await asyncio.gather(kachaka.human_detection())  # Detect human
    await asyncio.gather(kachaka.face_detector.process(kachaka.cv_img))

    # Stalking stage
    if kachaka.target_found:
        # cancel navigation if any and prevent future navigations
        if kachaka.cd > DURATION_FOR_CANCEL_NAV:
            if await kachaka.check_navigate():
                await kachaka.async_client.cancel_command()
            kachaka.run_nav = False
        print(f"ID:{kachaka.id} | .cd:{kachaka.cd} | .face_found_count:{kachaka.face_found_count} | .run_nav:{kachaka.run_nav} | (.linear,.angular):({round(kachaka.linear,3)},{round(kachaka.angular,3)})")
        kachaka.cd += 1
        if kachaka.cd > SEARCH_COOLDOWN:
            await kachaka.adjust()
            await kachaka.move()
            if kachaka.face_detector.is_face_present():
                kachaka.face_detector.draw_landmarks(kachaka.cv_img)
                kachaka.face_found_count += 1
            else:
                kachaka.face_found_count -= 1

            if not(0 < kachaka.face_found_count <= SEARCH_PERIOD):
                kachaka.target_found = False
                kachaka.face_found_count = SEARCH_PERIOD//2
                kachaka.cd = 0
                if kachaka.face_found_count > SEARCH_PERIOD:
                    print("human found")
                    await kachaka.speak("こんにちは！")
                elif kachaka.face_found_count <= 0:
                    print("target lost")
                    await kachaka.speak("顔が見つかりませんでした")
    else:
        kachaka.face_found_count = SEARCH_PERIOD//2
        kachaka.run_nav = True
        kachaka.cd = 0

    # Annotation task
    await kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True)

async def controller(kachakas:list[KachakaFrame]):
    """
    asynchronously runs navigation on each kachaka and detection_tasks then displays
    navigation() only runs if kachaka object has not found a target
    """
    print(f"{C.GREEN}Loaded{C.RESET} controller()")
    while any([kachaka.run for kachaka in kachakas]):
        await asyncio.gather(
            *[asyncio.create_task(detection_process(kachaka)) for i,kachaka in enumerate(kachakas)]
        )
        await display_kachakas(kachakas)

async def navigator(kachaka:KachakaFrame):
    while kachaka.run:
        nav_running = await kachaka.check_navigate()
        if kachaka.run_nav:
            if not nav_running:
                await kachaka.short_navigate()
        else:
            await kachaka.cancel_navigation()

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]
    move_tasks = [asyncio.create_task(kachaka.move()) for kachaka in kachakas]
    navigate_tasks = [asyncio.create_task(navigator(kachaka)) for kachaka in kachakas]
    monitor_task = asyncio.create_task(object_monitor_key_press(kachakas))
    controller_task = asyncio.create_task(controller(kachakas))

    print(f"{C.BLUE}Starting{C.RESET} Script")
    await asyncio.gather(
        controller_task,
        *move_tasks,
        *navigate_tasks,
        monitor_task
        )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())