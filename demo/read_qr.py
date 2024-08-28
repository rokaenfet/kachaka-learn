import time
import cv2
import kachaka_api
import numpy as np
import asyncio
from IPython.display import Image, clear_output, display

KACHAKA_IP = "192.168.118.158:26400"

# qr text-code generator https://pf-robotics.github.io/textcode/
async def read_qr(client:kachaka_api.aio.KachakaApiClient):
    qcd = cv2.QRCodeDetector()
    async for image in client.front_camera_ros_compressed_image.stream():
        cv_image = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        decoded_info, corners, _ = qcd.detectAndDecode(cv_image)
        if corners is not None:
            cv_image = cv2.polylines(cv_image, corners.astype(int), True, (0, 0, 255), 2)
        if decoded_info != "":
            cv_image = cv2.putText(
                cv_image,
                decoded_info,
                corners[0][0].astype(int),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        _, ret = cv2.imencode(
            ".jpg",
            cv2.resize(cv_image, (int(cv_image.shape[1] / 2), int(cv_image.shape[0] / 2))),
        )
        clear_output(wait=True)
        cv2.imshow("",cv_image)
        cv2.waitKey(1)

async def main():
    client = kachaka_api.aio.KachakaApiClient(KACHAKA_IP)
    await client.set_auto_homing_enabled(False)
    await read_qr(client)

if __name__ == "__main__":
    asyncio.run(main())