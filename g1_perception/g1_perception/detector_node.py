"""Instrument detector node -> /g1/perception/detections (std_msgs/String JSON, contract 6).

    ros2 run g1_perception detector_node --ros-args -p mode:=gt      # ground truth from /sim/objects (sim only)
    ros2 run g1_perception detector_node --ros-args -p mode:=onnx -p model:=<path>.onnx

Message: [{"label", "score", "bbox_obb": [cx, cy, w, h, angle_deg] (image pixels, long side angle) | null,
           "pose": {"frame", "xyz", "yaw"},            # instrument frame (= /sim/objects frame: tip-end origin)
           "grasp": {"frame", "xyz", "yaw", "upright"},  # grasp point from OBB + depth (onnx) / catalogue site (gt)
           "source": "onnx" | "ground_truth"}]
mode gt   : CHEAT for the sim. Every instrument of /sim/objects, score 1, "source": "ground_truth"; bbox_obb projected
            from the catalogue box when the camera pose is known. Needs no camera images.
mode onnx : an external YOLO-OBB ONNX export (see obb_onnx.py for the contract) on /g1/head_camera/image_raw + depth + camera_info; pose and
            grasp from g1_perception.grasp (OBB + aligned depth). Frame = output_frame when the camera pose is known
            (TF, or /sim/ground_truth + TF in the sim), else the camera's d435_link frame.
Also publishes /g1/perception/annotated/compressed (JPEG with the boxes) at <= 5 Hz for the UI.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from .camera_pose import CameraPose
from .geometry import R_LINK_OPT
from .grasp import grasp_from_obb
from .gt import objects_to_detections

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_perception/g1_perception/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", ".."))


def image_to_np(msg):
    if msg.encoding in ("rgb8", "bgr8"):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3 if msg.step else msg.width, 3)[:, :msg.width]
        return a[..., ::-1] if msg.encoding == "bgr8" else a
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    raise ValueError(f"unsupported encoding {msg.encoding}")


class DetectorNode(Node):
    def __init__(self):
        super().__init__("g1_perception_detector")
        p = self.declare_parameter
        self.mode = p("mode", "gt").value
        # ONNX detector: ROS parameter `model`, else $G1_PERCEPTION_MODEL. No default inside the stack.
        model = p("model", os.environ.get("G1_PERCEPTION_MODEL", "")).value
        self.catalog = json.load(open(p("catalog", os.path.join(ROOT, "hospital-sim-env/objects/catalog.json")).value))
        self.output_frame = p("output_frame", "map").value
        self.camera_frame = p("camera_frame", "d435_link").value
        self.max_hz = float(p("max_hz", 10.0).value)
        self.conf = float(p("conf", 0.35).value)
        self.annotate = bool(p("annotate", True).value)
        self.cam = CameraPose(self, self.output_frame, self.camera_frame)
        self.pub = self.create_publisher(String, "/g1/perception/detections", 10)
        self.pub_ann = self.create_publisher(CompressedImage, "/g1/perception/annotated/compressed", 2)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.K, self.size, self.depth = None, None, None
        self.create_subscription(CameraInfo, "/g1/head_camera/camera_info", self._info, qos)
        self.create_subscription(Image, "/g1/head_camera/depth", self._depth, qos)
        self.create_subscription(Image, "/g1/head_camera/image_raw", self._image, qos)
        self.last = 0.0
        self.n, self.t_stat, self.infer_ms = 0, time.monotonic(), []
        if self.mode == "gt":
            self.get_logger().warn("mode=gt: detections are GROUND TRUTH from /sim/objects (sim only, flagged "
                                   "'source': 'ground_truth')")
            self.create_subscription(String, "/sim/objects", self._objects, 10)
            self.det = None
        else:
            from .obb_onnx import OBBDetector

            if not model or not os.path.exists(model):
                raise SystemExit("mode=onnx needs a detector: set the `model` parameter or G1_PERCEPTION_MODEL "
                                 f"to an ONNX file (got {model!r}). Use mode=gt in simulation.")
            self.det = OBBDetector(model, conf=self.conf)
            self.get_logger().info(f"mode=onnx: {model}, labels {self.det.labels}")
        self.create_timer(5.0, self._stats)

    def _info(self, m):
        self.K = np.array(m.k, float).reshape(3, 3)
        self.size = (m.width, m.height)

    def _depth(self, m):
        self.depth = image_to_np(m)

    def _objects(self, m):
        now = time.monotonic()
        if now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        objs = json.loads(m.data)
        T = self.cam.T_out_opt() if self.K is not None else None
        dets = objects_to_detections(objs, self.catalog, T, self.K, self.size)
        self.pub.publish(String(data=json.dumps(dets)))
        self.n += 1

    def _image(self, m):
        if self.det is None:  # gt mode: images are not needed
            return
        now = time.monotonic()
        if now - self.last < 1.0 / self.max_hz:
            return
        self.last = now
        rgb = image_to_np(m)
        t0 = time.perf_counter()
        raw = self.det(rgb)
        self.infer_ms.append(1000 * (time.perf_counter() - t0))
        T = self.cam.T_out_opt()
        frame = self.output_frame
        if T is None:  # no camera pose: express results in d435_link
            T = np.eye(4)
            T[:3, :3] = R_LINK_OPT
            frame = self.camera_frame
        dets = []
        for label, score, obb in raw:
            d = {"label": label, "score": round(score, 3), "bbox_obb": [round(v, 2) for v in obb], "source": "onnx"}
            if self.depth is not None and self.K is not None and self.depth.shape[:2] == rgb.shape[:2]:
                L = self.catalog["instruments"].get(label, {}).get("size_xyz_m", [0, 0, None])[2]
                g = grasp_from_obb(obb, self.depth, self.K, T, length=L)
                if g:
                    d["grasp"] = {"frame": frame, "xyz": [round(v, 4) for v in g["grasp_xyz"]], "yaw": round(g["grasp_yaw"], 4),
                                  "upright": g["upright"], "axis": [round(v, 3) for v in g["axis"]]}
                    xyz = g["origin_xyz"] or g["grasp_xyz"]
                    d["pose"] = {"frame": frame, "xyz": [round(v, 4) for v in xyz], "yaw": round(g["grasp_yaw"], 4)}
            dets.append(d)
        self.pub.publish(String(data=json.dumps(dets)))
        self.n += 1
        if self.annotate and self.pub_ann.get_subscription_count() > 0:
            self._publish_annotated(rgb, dets, m.header)

    def _publish_annotated(self, rgb, dets, header):
        import cv2

        img = np.ascontiguousarray(rgb[..., ::-1])
        for d in dets:
            cx, cy, w, h, a = d["bbox_obb"]
            pts = cv2.boxPoints(((cx, cy), (w, h), a)).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(img, f"{d['label']} {d['score']:.2f}", tuple(int(v) for v in pts.min(0)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            msg = CompressedImage()
            msg.header, msg.format, msg.data = header, "jpeg", buf.tobytes()
            self.pub_ann.publish(msg)

    def _stats(self):
        el = time.monotonic() - self.t_stat
        inf = f", inference {np.mean(self.infer_ms):.1f} ms" if self.infer_ms else ""
        self.get_logger().info(f"{self.mode}: {self.n / el:.1f} Hz detections{inf}, camera pose: {self.cam.source}")
        self.n, self.t_stat, self.infer_ms = 0, time.monotonic(), []


def main():
    rclpy.init()
    n = DetectorNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
