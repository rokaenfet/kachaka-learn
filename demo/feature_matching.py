import time
import cv2
import kachaka_api
import numpy as np
import asyncio
from IPython.display import Image, clear_output, display

KACHAKA_IP = "192.168.118.158:26400"

async def feature_matching(client: kachaka_api.aio.KachakaApiClient):
    target_area_length = 400
    time_to_capture = 10

    start_time = time.time()
    async for image in client.front_camera_ros_compressed_image.stream():
        cv_image = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        start_x = int(cv_image.shape[1] / 2 - target_area_length / 2)
        start_y = int(cv_image.shape[0] / 2 - target_area_length / 2)
        end_x = int(cv_image.shape[1] / 2 + target_area_length / 2)
        end_y = int(cv_image.shape[0] / 2 + target_area_length / 2)

        remaining_time = time_to_capture - (time.time() - start_time)
        text = f"{remaining_time:.2f} [sec]" if remaining_time > 0 else "capture!"

        cv_image = cv2.rectangle(
            cv_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), thickness=2
        )
        cv_image = cv2.putText(
            cv_image,
            text,
            (start_x, start_y - 10),
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
        cv2.imshow("", cv_image)
        cv2.waitKey(1)
        if remaining_time <= 0.0:
            break

    orig_image = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)[
        start_y:end_y, start_x:end_x
    ]
    orb = cv2.ORB_create()
    orig_keypoints, orig_descriptors = orb.detectAndCompute(orig_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ratio_threshold = 0.7

    async for image in client.front_camera_ros_compressed_image.stream():
        cv_image = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        keypoints, descriptors = orb.detectAndCompute(cv_image, None)
        matches = bf.knnMatch(orig_descriptors, descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good.append([m])
        cv_image = cv2.drawMatchesKnn(
            orig_image,
            orig_keypoints,
            cv_image,
            keypoints,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv_image = cv2.putText(
            cv_image,
            f"good matches: {len(good)}",
            (50, cv_image.shape[0] - 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        clear_output(wait=True)
        _, ret = cv2.imencode(
            ".jpg",
            cv2.resize(cv_image, (int(cv_image.shape[1] / 2), int(cv_image.shape[0] / 2))),
        )
        cv2.imshow("", cv_image)
        cv2.waitKey(1)

async def main():
    client = kachaka_api.aio.KachakaApiClient(f"{KACHAKA_HOST}:{KACHAKA_PORT}")
    await client.set_auto_homing_enabled(False)
    await feature_matching(client)

if __name__ == "__main__":
    asyncio.run(main())