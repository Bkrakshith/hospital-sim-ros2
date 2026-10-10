"""Surgeon hand -> /g1/perception/surgeon_hand (geometry_msgs/PoseStamped, contract 6).

    ros2 run g1_perception surgeon_hand_node --ros-args -p mode:=gt          # /sim/objects "surgeon_hand" (sim CHEAT)
    ros2 run g1_perception surgeon_hand_node --ros-args -p mode:=mediapipe   # head camera RGB-D + MediaPipe hands

mediapipe: HandLandmarker (the model g1-webcam-teleop/xr_teleoperate/teleop/webcam uses, models/hand_landmarker.task) on
/g1/head_camera/image_raw at <= max_hz. The palm centre is the mean of landmarks 0, 5, 9, 13, 17 (wrist + finger
bases); its depth is the median of the aligned depth in a 9x9 window; back-projected with K and expressed in
output_frame (TF, or /sim/ground_truth + TF in the sim) or, without a camera pose, in d435_link. With several hands
the one with the highest handedness score whose palm is 0.3-2.5 m away wins. Orientation: identity (the handover
skill uses the position). The sim's surgeon mannequin is primitive geometry, which MediaPipe does not detect: use
mode gt in the sim until the surgeon is a rigged human.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .camera_pose import CameraPose
from .detector_node import image_to_np
from .geometry import R_LINK_OPT, backproject

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_perception/g1_perception/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", ".."))
PALM = (0, 5, 9, 13, 17)


def palm_pixel(landmarks, W, H):
    pts = np.array([[landmarks[i].x * W, landmarks[i].y * H] for i in PALM])
    return pts.mean(0)


def depth_at(depth, u, v, r=4):
    H, W = depth.shape
    u0, v0 = int(round(u)), int(round(v))
    win = depth[max(0, v0 - r):min(H, v0 + r + 1), max(0, u0 - r):min(W, u0 + r + 1)].ravel()
    win = win[np.isfinite(win) & (win > 0.05)]
    return float(np.median(win)) if win.size else None


class SurgeonHandNode(Node):
    def __init__(self):
        super().__init__("g1_perception_surgeon_hand")
        p = self.declare_parameter
        self.mode = p("mode", "gt").value
        self.output_frame = p("output_frame", "map").value
        self.max_hz = float(p("max_hz", 10.0).value)
        model = p("model", os.path.join(ROOT, "g1-webcam-teleop/xr_teleoperate/teleop/webcam/models/hand_landmarker.task")).value
        self.pub = self.create_publisher(PoseStamped, "/g1/perception/surgeon_hand", 10)
        self.last, self.n, self.t_stat = 0.0, 0, time.monotonic()
        if self.mode == "gt":
            self.get_logger().warn("mode=gt: surgeon hand is GROUND TRUTH from /sim/objects (sim only)")
            self.create_subscription(String, "/sim/objects", self._objects, 10)
        else:
            from mediapipe.tasks.python import BaseOptions, vision

            self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model, delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, num_hands=4, min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
            self.cam = CameraPose(self, self.output_frame)
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.K, self.depth, self.ts = None, None, 0
            self.create_subscription(CameraInfo, "/g1/head_camera/camera_info", self._info, qos)
            self.create_subscription(Image, "/g1/head_camera/depth", self._depth, qos)
            self.create_subscription(Image, "/g1/head_camera/image_raw", self._image, qos)
        self.create_timer(5.0, self._stats)

    def _objects(self, m):
        now = time.monotonic()
        if now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        o = json.loads(m.data).get("surgeon_hand")
        if not o:
            return
        x, y, z, qw, qx, qy, qz = o["pose"]
        self._publish("map", (x, y, z), (qw, qx, qy, qz))

    def _info(self, m):
        self.K = np.array(m.k, float).reshape(3, 3)

    def _depth(self, m):
        self.depth = image_to_np(m)

    def _image(self, m):
        now = time.monotonic()
        if now - self.last < 1.0 / self.max_hz or self.K is None or self.depth is None:
            return
        self.last = now
        import mediapipe as mp

        rgb = np.ascontiguousarray(image_to_np(m))
        self.ts += max(1, int(1000 / self.max_hz))
        res = self.hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.ts)
        H, W = rgb.shape[:2]
        best = None
        for lm, hd in zip(res.hand_landmarks, res.handedness):
            u, v = palm_pixel(lm, W, H)
            z = depth_at(self.depth, u, v)
            if z is None or not (0.3 < z < 2.5):
                continue
            score = hd[0].score if hd else 0.0
            if best is None or score > best[0]:
                best = (score, backproject(u, v, z, self.K))
        if best is None:
            return
        T = self.cam.T_out_opt()
        if T is None:
            self._publish("d435_link", R_LINK_OPT @ best[1], (1.0, 0.0, 0.0, 0.0))
        else:
            self._publish(self.output_frame, T[:3, :3] @ best[1] + T[:3, 3], (1.0, 0.0, 0.0, 0.0))

    def _publish(self, frame, xyz, q):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in xyz)
        msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z = (float(v) for v in q)
        self.pub.publish(msg)
        self.n += 1

    def _stats(self):
        el = time.monotonic() - self.t_stat
        self.get_logger().info(f"{self.mode}: {self.n / el:.1f} Hz surgeon_hand")
        self.n, self.t_stat = 0, time.monotonic()


def main():
    rclpy.init()
    n = SurgeonHandNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
