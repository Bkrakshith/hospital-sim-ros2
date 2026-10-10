"""Where is the head camera? Shared by the detector and surgeon-hand nodes.

Order: (1) TF output_frame <- camera_frame (map -> ... -> d435_link, when SLAM/AMCL + robot_state_publisher run);
(2) sim fallback: /sim/ground_truth (pelvis in the world = map frame of the sims) composed with TF pelvis <- d435_link;
(3) nothing: callers publish in the camera frame instead. Returns T (4x4) of the OPTICAL camera frame in the output
frame (d435_link rotated by geometry.R_LINK_OPT).
"""
from __future__ import annotations

import numpy as np

from .geometry import R_LINK_OPT, make_T, quat_wxyz_to_R


def tf_to_T(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return make_T(quat_wxyz_to_R([q.w, q.x, q.y, q.z]), [t.x, t.y, t.z])


class CameraPose:
    def __init__(self, node, output_frame="map", camera_frame="d435_link", base_frame="pelvis", use_ground_truth=True):
        import tf2_ros
        from nav_msgs.msg import Odometry

        self.node, self.output_frame, self.camera_frame, self.base_frame = node, output_frame, camera_frame, base_frame
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, node)
        self.gt = None
        if use_ground_truth:
            node.create_subscription(Odometry, "/sim/ground_truth", self._gt, 10)
        self.source = None

    def _gt(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.gt = make_T(quat_wxyz_to_R([q.w, q.x, q.y, q.z]), [p.x, p.y, p.z])

    def T_out_opt(self):
        from rclpy.time import Time

        T_link_opt = make_T(R_LINK_OPT, [0, 0, 0])
        try:
            tf = self.buf.lookup_transform(self.output_frame, self.camera_frame, Time())
            self.source = "tf"
            return tf_to_T(tf) @ T_link_opt
        except Exception:  # noqa: BLE001
            pass
        if self.gt is not None:
            try:
                tf = self.buf.lookup_transform(self.base_frame, self.camera_frame, Time())
                self.source = "ground_truth+tf"
                return self.gt @ tf_to_T(tf) @ T_link_opt
            except Exception:  # noqa: BLE001
                pass
        self.source = None
        return None
