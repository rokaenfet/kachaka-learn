import cv2
import asyncio
import keyboard

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    # 1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

async def process(kachaka:KachakaFrame):
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
        kachaka.angular, kachaka.linear = 0.5, 0
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
    if kachaka.face_detector.is_face_present():
        kachaka.face_detector.draw_landmarks(kachaka.cv_img)
    await kachaka.emergency_stop()
    await kachaka.move()
    kachaka.annotate(st, show_fps=True, show_nearest_lidar=False, show_id=True) # annotate img

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]

    while True:
        await asyncio.gather(
            *(process(kachaka) for kachaka in kachakas),
        )
        await display_kachakas(kachakas)
        if keyboard.is_pressed("q"):
            break
        await asyncio.sleep(0.00001)  # prevent busy loop

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())