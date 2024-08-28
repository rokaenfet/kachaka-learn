import io
from math import atan2, radians

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import kachaka_api
import asyncio
import cv2
from kachaka_api.util.geometry import MapImage2DGeometry
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
from PIL import Image

KACHAKA_IP = "192.168.118.158:26400"
# ros2/kachaka_description/robot/kachaka.urdf for size ref
ROBOT_SIZE_X = 0.387
ROBOT_SIZE_Y = 0.24
BASE_FOOTPRINT_TO_BODY_RECT_ORIGIN = Affine2D().translate(-0.15, -ROBOT_SIZE_Y / 2)
BASE_FOOTPRINT_TO_LASER_FRAME = Affine2D().rotate(radians(90)).translate(0.156, 0)
last_target = None

def draw_robot(ax, fig_origin_to_base_footprint):
    # draw body
    return [
        ax.add_patch(
            patches.Rectangle(
                (0, 0),
                ROBOT_SIZE_X,
                ROBOT_SIZE_Y,
                facecolor="gray",
                transform=BASE_FOOTPRINT_TO_BODY_RECT_ORIGIN
                + fig_origin_to_base_footprint,
            )
        ),
        # draw LED ring
        ax.add_patch(
            patches.Circle(
                (0, 0),
                radius=0.045,
                facecolor="gray",
                edgecolor="white",
                transform=BASE_FOOTPRINT_TO_LASER_FRAME + fig_origin_to_base_footprint,
            )
        ),
    ]


def draw_scan(ax, fig_origin_to_base_footprint, scan):
    theta = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    dist = np.array(scan.ranges)
    return ax.scatter(
        dist * np.cos(theta),
        dist * np.sin(theta),
        c="red",
        s=1,
        transform=BASE_FOOTPRINT_TO_LASER_FRAME + fig_origin_to_base_footprint,
    )

def main():
    client = kachaka_api.KachakaApiClient(KACHAKA_IP)
    client.set_auto_homing_enabled(False)

    png_map = client.get_png_map()
    png_map_img = Image.open(io.BytesIO(png_map.data))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    drawn_artists = []

    map_image_2d_geometry = MapImage2DGeometry(png_map)

    def onclick(event):
        global last_target
        if last_target:
            last_target.remove()
        last_target = ax.scatter(event.xdata, event.ydata)
        # clickされた位置に移動
        pixel_xy = (event.xdata, event.ydata)
        angle = 0
        pose_mat = map_image_2d_geometry.calculate_robot_pose_matrix_from_pixel(
            pixel_xy, angle
        )
        client.move_to_pose(
            x=pose_mat[0, 2],
            y=pose_mat[1, 2],
            yaw=atan2(pose_mat[1][0], pose_mat[0][0]),
            wait_for_completion=False,
        )

    def update_plot(frame):
        while drawn_artists:
            drawn_artists.pop().remove()

        # ロボットを描画
        robot_pose = client.get_robot_pose()
        image_origin_to_robot = Affine2D(
            map_image_2d_geometry.calculate_robot_pose_matrix_in_pixel(robot_pose)
        )
        robot_artists = draw_robot(ax, image_origin_to_robot + ax.transData)
        # scanを描画
        scan = client.get_ros_laser_scan()
        scan_artist = draw_scan(ax, image_origin_to_robot + ax.transData, scan)

        drawn_artists.extend(robot_artists)
        drawn_artists.append(scan_artist)

    ax.imshow(png_map_img)
    fig.canvas.mpl_connect("button_press_event", onclick)
    # 60秒間ロボットを描画する
    func_animation = FuncAnimation(fig, update_plot, interval=100, frames=600, repeat=False)
    # func_animation.save("demo/move_robot.mp4")
    plt.show()

if __name__ == "__main__":
    main()