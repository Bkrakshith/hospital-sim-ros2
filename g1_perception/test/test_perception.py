"""Unit tests for g1_perception (pure python; no ROS graph needed).

    cd ros2_ws/src/hospital-sim-ros2/g1_perception && python -m pytest -q test/      (after `source env.sh`)
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_perception.geometry import R_LINK_OPT, make_T, min_area_rect, project, quat_wxyz_to_R  # noqa: E402
from g1_perception.grasp import grasp_from_obb  # noqa: E402
from g1_perception.gt import objects_to_detections  # noqa: E402
from g1_perception.obb_onnx import decode, letterbox, probiou, regularize  # noqa: E402

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_perception/test/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
CAT = json.load(open(os.path.join(ROOT, "hospital-sim-env/objects/catalog.json")))
K = np.array([[468.9, 0, 320], [0, 468.9, 180], [0, 0, 1.0]])


def test_probiou_identity_and_disjoint():
    b = np.array([[100, 100, 80, 10, 0.3]])
    assert probiou(b, b)[0, 0] > 0.99
    far = np.array([[400, 300, 80, 10, 0.3]])
    assert probiou(b, far)[0, 0] < 0.01


def test_decode_nms_keeps_one_per_object():
    nc, N = 7, 6
    raw = np.zeros((4 + nc + 1, N), np.float32)
    boxes = [[100, 100, 80, 12, 0.2], [101, 99, 79, 12, 0.21], [400, 200, 60, 20, 1.0], [0, 0, 5, 5, 0], [0, 0, 5, 5, 0], [0, 0, 5, 5, 0]]
    for i, b in enumerate(boxes):
        raw[:4, i] = b[:4]
        raw[4 + nc, i] = b[4]
    raw[4 + 3, 0], raw[4 + 3, 1], raw[4 + 6, 2] = 0.9, 0.8, 0.7
    dets = decode(raw, nc, conf=0.25)
    assert [d[0] for d in dets] == [3, 6]
    assert dets[0][1] == pytest.approx(0.9)


def test_regularize_long_side():
    cx, cy, w, h, a = regularize([0, 0, 10, 40, 0.1])
    assert w == 40 and h == 10 and 0 <= a < math.pi and a == pytest.approx(0.1 + math.pi / 2)


def test_letterbox_shapes():
    img = np.zeros((360, 640, 3), np.uint8)
    x, r, (px, py) = letterbox(img, 640)
    assert x.shape == (1, 3, 640, 640) and r == pytest.approx(1.0) and px == 0 and py == 140


def _scene_depth(rod_upright=True):
    """Camera 1 m above a table plane (looking down), a 1 cm rod standing up / lying at the image centre."""
    H, W = 360, 640
    depth = np.full((H, W), 1.0, np.float32)  # table at 1.0 m along the optical axis
    if rod_upright:  # seen from above an upright rod is a small disc, top 15 cm closer
        yy, xx = np.mgrid[:H, :W]
        depth[(xx - 320) ** 2 + (yy - 180) ** 2 < 5 ** 2] = 0.85
        obb = [320, 180, 12, 10, 0.0]
    else:  # lying rod along image x, 12 cm long, 1 cm thick, top 1 cm above the table
        depth[176:185, 260:380] = 0.99
        obb = [320, 180, 124, 12, 0.0]
    return depth, obb


def test_grasp_lying_rod_yaw_and_position():
    depth, obb = _scene_depth(False)
    # camera looking straight down at z = 1 m above a table at z = 0 (optical z -> world -z, x -> world x, y -> -y)
    T = make_T(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]), [0, 0, 1.0])
    g = grasp_from_obb(obb, depth, K, T, length=0.12)
    assert g and not g["upright"]
    assert np.allclose(g["grasp_xyz"], [0, 0, 0.01], atol=0.004)
    assert abs(math.sin(g["grasp_yaw"])) < 0.05  # long axis along world x (mod pi)


def test_grasp_upright_rod_from_side():
    # side view: camera 0.5 m in front of a vertical rod (optical z = world x, optical y = -world z)
    H, W = 360, 640
    depth = np.full((H, W), 1.5, np.float32)
    depth[100:260, 315:326] = 0.5
    T = make_T(np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]]), [0, 0, 1.0])
    g = grasp_from_obb([320, 180, 160, 11, 90.0], depth, K, T, length=0.2)
    assert g["upright"]
    assert g["grasp_xyz"][0] == pytest.approx(0.5, abs=0.01)
    assert abs(g["grasp_yaw"]) < 0.05  # approach heading = camera -> rod = +x
    top = 1.0 + (180 - 100) * 0.5 / 468.9
    assert g["origin_xyz"][2] == pytest.approx(top - 0.2, abs=0.01)


def test_gt_detections_contract_and_projection():
    wp = CAT["world_poses"]
    objs = {n: {"pose": list(wp[n]["pos"]) + list(wp[n]["quat"]), "type": "instrument"} for n in CAT["labels"]}
    objs["tool_rack"] = {"pose": list(wp["tool_rack"]["pos"]) + list(wp["tool_rack"]["quat"]), "type": "furniture"}
    # head camera at the table stance (pelvis 0.33 m from the rack row, head 1.258 m, pitched 47.6 deg down)
    x, y, yaw = CAT["robot_stances"]["theatre_instrument_table"]
    c, s = math.cos(yaw), math.sin(yaw)
    Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = math.radians(47.6)
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    T_map_opt = make_T(Rb @ Ry @ R_LINK_OPT, [x + 0.054 * c, y + 0.054 * s, 1.258])
    dets = objects_to_detections(objs, CAT, T_map_opt, K, (640, 360))
    assert sorted(d["label"] for d in dets) == sorted(CAT["labels"])
    for d in dets:
        assert set(d) >= {"label", "score", "bbox_obb", "pose", "source"} and d["source"] == "ground_truth"
        assert d["pose"]["frame"] == "map" and len(d["pose"]["xyz"]) == 3
        assert len(d["bbox_obb"]) == 5
    assert sum(d["in_view"] for d in dets) >= 5  # the rack is in front of the camera
    json.dumps(dets)


def test_min_area_rect_long_side():
    pts = np.array([[0, 0], [100, 0], [100, 10], [0, 10]], float)
    r = min_area_rect(pts)
    assert r[2] == pytest.approx(100, abs=1) and r[3] == pytest.approx(10, abs=1)


def test_onnx_detector_runs_if_model_present():
    model = os.path.join(ROOT, "hospital-sim-perception/models/instrument_obb_smoke.onnx")
    img_path = os.path.join(ROOT, "hospital-sim-perception/data/sample_v1/rgb/000000.jpg")
    if not (os.path.exists(model) and os.path.exists(img_path)):
        pytest.skip("no model / sample data")
    from PIL import Image

    from g1_perception.obb_onnx import OBBDetector

    det = OBBDetector(model, conf=0.01)
    out = det(np.asarray(Image.open(img_path).convert("RGB")))
    assert isinstance(out, list)
    for label, score, obb in out:
        assert label in CAT["labels"] and 0 <= score <= 1 and len(obb) == 5 and obb[2] >= obb[3]
