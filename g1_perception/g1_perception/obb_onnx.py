"""YOLO11-OBB ONNX inference with onnxruntime + numpy post-processing (no torch / ultralytics at run time).

Model contract: a static 1x3xSxS YOLO-OBB ONNX export with a sidecar <model>.json ({"labels": [...], "imgsz": S}) and
the raw head [1, 4 + nc + 1, N]: cx, cy, w, h in input pixels, nc sigmoid class scores, angle in rad. Post-processing: score threshold, class-aware rotated NMS using the
probabilistic IoU of the boxes' Gaussian approximations (as Ultralytics), undo the letterbox.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def letterbox(img, size):
    """RGB uint8 HxWx3 -> float32 1x3xSxS (0..1), scale, (pad_x, pad_y)."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    px, py = (size - nw) / 2, (size - nh) / 2
    out = np.full((size, size, 3), 114, np.uint8)
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out[top:top + nh, left:left + nw] = rs
    x = out.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, (left, top)


def _cov(b):
    """Gaussian covariance terms (a, b, c) of rotated boxes [cx, cy, w, h, angle]."""
    w2, h2 = b[:, 2] ** 2 / 12, b[:, 3] ** 2 / 12
    c, s = np.cos(b[:, 4]), np.sin(b[:, 4])
    return w2 * c ** 2 + h2 * s ** 2, w2 * s ** 2 + h2 * c ** 2, (w2 - h2) * c * s


def probiou(b1, b2, eps=1e-7):
    """Pairwise probabilistic IoU (Bhattacharyya-based) between two sets of rotated boxes -> [n1, n2]."""
    x1, y1 = b1[:, 0:1], b1[:, 1:2]
    x2, y2 = b2[None, :, 0], b2[None, :, 1]
    a1, bb1, c1 = (v[:, None] for v in _cov(b1))
    a2, bb2, c2 = (v[None, :] for v in _cov(b2))
    den = (a1 + a2) * (bb1 + bb2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (bb1 + bb2) * (x1 - x2) ** 2) / den * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / den * 0.5
    t3 = np.log(den / (4 * np.sqrt(np.clip(a1 * bb1 - c1 ** 2, 0, None) * np.clip(a2 * bb2 - c2 ** 2, 0, None)) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def rotated_nms(boxes, scores, iou_thr=0.45):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou = probiou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou < iou_thr]
    return np.array(keep, int)


def decode(raw, nc, conf=0.25, iou=0.45, max_det=50):
    """raw [4 + nc + 1, N] -> list of (cls, score, [cx, cy, w, h, angle_rad]) in letterboxed input pixels."""
    raw = np.asarray(raw)
    boxes = raw[:4].T
    cls_scores = raw[4:4 + nc].T
    ang = raw[4 + nc]
    cls = cls_scores.argmax(1)
    sc = cls_scores[np.arange(len(cls)), cls]
    m = sc > conf
    if not m.any():
        return []
    b = np.concatenate([boxes[m], ang[m, None]], 1)
    sc, cls = sc[m], cls[m]
    out = []
    for c in np.unique(cls):  # class-aware NMS
        idx = np.where(cls == c)[0]
        for k in rotated_nms(b[idx], sc[idx], iou):
            out.append((int(c), float(sc[idx][k]), b[idx][k].tolist()))
    out.sort(key=lambda t: -t[1])
    return out[:max_det]


def regularize(obb):
    """[cx, cy, w, h, angle_rad] -> w >= h, angle in [0, pi) (long side direction)."""
    cx, cy, w, h, a = obb
    if w < h:
        w, h, a = h, w, a + math.pi / 2
    return [cx, cy, w, h, a % math.pi]


class OBBDetector:
    def __init__(self, onnx_path, conf=0.35, iou=0.45, providers=None):
        import onnxruntime as ort

        p = Path(onnx_path)
        side = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
        self.labels = side.get("labels")
        self.imgsz = int(side.get("imgsz", 640))
        self.sess = ort.InferenceSession(str(p), providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        out_shape = self.sess.get_outputs()[0].shape
        self.nc = len(self.labels) if self.labels else int(out_shape[1]) - 5
        self.conf, self.iou = conf, iou

    def __call__(self, rgb):
        """RGB uint8 image -> [(label, score, [cx, cy, w, h, angle_deg])] in image pixels (angle of the long side)."""
        x, r, (px, py) = letterbox(rgb, self.imgsz)
        raw = self.sess.run(None, {self.inp: x})[0][0]
        dets = []
        for c, s, (cx, cy, w, h, a) in decode(raw, self.nc, self.conf, self.iou):
            cx, cy, w, h, a = regularize([(cx - px) / r, (cy - py) / r, w / r, h / r, a])
            dets.append((self.labels[c] if self.labels else str(c), s, [cx, cy, w, h, math.degrees(a)]))
        return dets
