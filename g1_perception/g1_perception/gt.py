"""Ground-truth "detections": /sim/objects -> the contract detection list (CLEARLY FLAGGED: "source": "ground_truth").

pose = the instrument's /sim/objects frame (tip-end origin, yaw = heading of its +x axis), frame "map" (sim world).
bbox_obb = min-area rectangle of the projected catalogue bounding box when a camera pose + K are given (else null),
in_view = the box centre projects inside the image in front of the camera. No occlusion reasoning.
"""
from __future__ import annotations

import numpy as np

from .geometry import box_corners, min_area_rect, pose7_to_T, project, yaw_of


def objects_to_detections(objects, catalog, T_map_cam=None, K=None, size=None, only_instruments=True):
    """objects: parsed /sim/objects {name: {"pose": [x,y,z,qw,qx,qy,qz], "type"}}; catalog: hospital-sim-env/objects/catalog.json;
    T_map_cam: 4x4 optical camera frame -> map (or None); K 3x3; size (W, H)."""
    dets = []
    inst = catalog.get("instruments", {})
    for name, o in objects.items():
        if only_instruments and o.get("type") != "instrument":
            continue
        T = pose7_to_T(o["pose"])
        det = {"label": name, "score": 1.0, "bbox_obb": None, "source": "ground_truth",
               "pose": {"frame": "map", "xyz": [round(float(v), 4) for v in T[:3, 3]], "yaw": round(float(yaw_of(T[:3, :3])), 4)}}
        c = inst.get(name)
        if c:
            gs = T @ np.r_[c["grasp"]["site"], 1.0]
            det["grasp"] = {"frame": "map", "xyz": [round(float(v), 4) for v in gs[:3]], "yaw": det["pose"]["yaw"]}
        if c and T_map_cam is not None and K is not None:
            Tcm = np.linalg.inv(T_map_cam) @ T
            P = box_corners(c["bbox_min"], c["bbox_max"]) @ Tcm[:3, :3].T + Tcm[:3, 3]
            if (P[:, 2] > 0.05).all():
                uv = project(P, K)
                det["bbox_obb"] = [round(v, 2) for v in min_area_rect(uv)]
                W, H = size or (1e9, 1e9)
                cx, cy = det["bbox_obb"][:2]
                det["in_view"] = bool(0 <= cx < W and 0 <= cy < H)
            else:
                det["in_view"] = False
        dets.append(det)
    return dets
