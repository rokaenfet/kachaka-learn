import cv2
import asyncio
import keyboard

from funcs import *

KACHAKA_IPS = {
    0:"192.168.118.158:26400",
    1:"192.168.118.159:26400",
    # 2:"192.168.118.77:26400"
    }

async def main():
    # initiate clients
    kachakas = [KachakaFrame(v, k) for k,v in KACHAKA_IPS.items()]

    while True:
        await asyncio.gather(
            *(kachaka.process_kachaka() for kachaka in kachakas),
        )
        await display_kachakas(kachakas)
        if keyboard.is_pressed("q"):
            break
        await asyncio.sleep(0.00001)  # prevent busy loop

    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())