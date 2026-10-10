#!/usr/bin/env python3
"""Save every head-camera frame to disk as JPEG (for a clean camera-view video), and republish a 1 Hz (sim time)
JPEG copy for rosbag2, so the bag carries the camera without gigabytes of raw images.

    camera_frame_saver.py --out <dir> [--topic /g1/head_camera/image_raw] [--quality 92] [--bag-hz 1]
Files: <dir>/<sim_ns>.jpg; <dir>/index.csv (sim_s, wall_s, file).
"""
import argparse
import io
import os
import time

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class Saver(Node):
    def __init__(self, a):
        super().__init__("camera_frame_saver")
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.idx = open(os.path.join(a.out, "index.csv"), "a")
        self.n = 0
        self.last_pub = -1e9
        self.pub = self.create_publisher(CompressedImage, a.topic + "/jpeg_1hz", 5)
        self.create_subscription(Image, a.topic, self.on_img, qos_profile_sensor_data)

    def on_img(self, m):
        if m.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "bgr8":
            a = a[..., ::-1]
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        buf = io.BytesIO()
        PILImage.fromarray(a).save(buf, format="JPEG", quality=self.a.quality)
        data = buf.getvalue()
        name = f"{m.header.stamp.sec:06d}{m.header.stamp.nanosec:09d}.jpg"
        with open(os.path.join(self.a.out, name), "wb") as f:
            f.write(data)
        self.idx.write(f"{t:.3f},{time.time():.3f},{name}\n")
        self.n += 1
        if self.n % 200 == 0:
            self.idx.flush()
            self.get_logger().info(f"{self.n} frames saved (sim t {t:.1f})")
        if t - self.last_pub >= 1.0 / self.a.bag_hz:
            self.last_pub = t
            c = CompressedImage()
            c.header = m.header
            c.format = "jpeg"
            c.data = data
            self.pub.publish(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="/g1/head_camera/image_raw")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--bag-hz", type=float, default=1.0)
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Saver(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.idx.close()


if __name__ == "__main__":
    main()
