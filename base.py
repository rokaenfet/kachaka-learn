import kachaka_api
import matplotlib.pyplot as plt
import numpy as np
# from IPython.display import Image, display
# ip changes everytime check with kachacar
KACHAKA_HOST = "192.168.118.158"
# grpc port
KACHAKA_PORT = 26400


def main():
    client = kachaka_api.KachakaApiClient(f"{KACHAKA_HOST}:{KACHAKA_PORT}")
    client.set_manual_control_enabled(True) # If true only moves with command

    # main velocity
    
    scan = client.get_ros_laser_scan()

    theta = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    dist = np.array(scan.ranges)

    # LiDARの点群を表示するサンプル
    plt.scatter(dist * np.cos(theta), dist * np.sin(theta))
    plt.savefig("test.png")


if __name__ == "__main__":
    main()