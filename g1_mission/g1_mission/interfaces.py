"""What skills see of the world. Every source of truth is behind one of these interfaces, so each step-1 CHEAT
(ground truth from the simulator) can be swapped for the real estimate without touching a skill.

| Interface          | Step-1 implementation (CHEAT)                  | Replacement                                         |
|--------------------|------------------------------------------------|-----------------------------------------------------|
| ObjectSource       | GroundTruthObjects  (/sim/objects)             | DetectionObjects (/g1/perception/detections)        |
| SurgeonHandSource  | GroundTruthSurgeonHand (/sim/objects entry)    | PerceptionSurgeonHand (/g1/perception/surgeon_hand) |
| grip detection     | sim weld flag "held_by" (proximity + dwell)    | hand force / tug detection or perception            |
| StationEstimator   | GroundTruthStation (/sim/ground_truth)         | table-marker (AprilTag/ArUco) or table-edge estimate|
| RobotInterface.base_T | /sim/ground_truth (fixed base / FineAlign)  | TF map -> pelvis (AMCL)                             |

All poses are 4x4 numpy matrices in the `map` frame (= sim world frame for the hospital_gt map).
"""
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from g1_mission.geometry import make_T, pose_to_T, rot_z, yaw_of, wrap


@dataclass
class Estimate:
    T: np.ndarray                 # 4x4, map frame
    stamp: float                  # seconds (wall clock of reception)
    source: str                   # "ground_truth", "perception", ...
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------- robot
class RobotInterface(ABC):
    """Proprioception + command channels (the /g1/auto/* topics on the robot, direct actuators offline)."""

    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def upper_q(self):
        """17 measured upper-body joint angles (waist 3, left arm 7, right arm 7) or None."""

    @abstractmethod
    def hand_q(self, side):
        """6 measured normalised hand positions (0 open .. 1 closed) or None."""

    @abstractmethod
    def base_T(self):
        """Pelvis pose in the map frame (4x4) or None."""

    @abstractmethod
    def command(self, upper=None, left_hand=None, right_hand=None, cmd_vel=None):
        """Send targets. upper: 17 joint targets; hands: 6 values 0..1; cmd_vel: (vx, vy, wz) body frame."""

    def head_image(self):
        """Latest head RGB image (H, W, 3 uint8) or None."""
        return None


# ---------------------------------------------------------------------------------------------------- objects
class ObjectSource(ABC):
    @abstractmethod
    def get(self, name):
        """Estimate of object `name` (object frame: tip at origin, +z to the handle) or None."""

    @abstractmethod
    def names(self): ...


class GroundTruthObjects(ObjectSource):
    """CHEAT: simulator ground truth. Feed it the /sim/objects JSON with update()."""

    def __init__(self):
        self.data, self.stamp = {}, 0.0

    def update(self, text_or_dict, stamp=None):
        self.data = json.loads(text_or_dict) if isinstance(text_or_dict, str) else dict(text_or_dict)
        self.stamp = time.time() if stamp is None else stamp

    def get(self, name):
        e = self.data.get(name)
        if not e or "pose" not in e:
            return None
        return Estimate(pose_to_T(e["pose"]), self.stamp, "ground_truth", {k: v for k, v in e.items() if k != "pose"})

    def names(self):
        return [n for n, e in self.data.items() if e.get("type") == "instrument"]


class DetectionObjects(ObjectSource):
    """Perception detections (/g1/perception/detections, contract 6): [{"label","score","bbox_obb","pose":{"frame",
    "xyz","yaw"}}]. Poses in frames other than map/world go through `to_map(frame, T) -> T or None` (the node passes a
    tf2 lookup). The instrument is assumed upright (yaw about +z) until the detector reports full 6-D poses."""

    def __init__(self, to_map=None, min_score=0.5):
        self.to_map, self.min_score = to_map, min_score
        self.best, self.stamp = {}, 0.0

    def update(self, text, stamp=None):
        dets = json.loads(text) if isinstance(text, str) else text
        best = {}
        for det in dets:
            if det.get("score", 1.0) < self.min_score or not det.get("pose"):
                continue
            p = det["pose"]
            T = make_T(rot_z(p.get("yaw", 0.0)), p["xyz"])
            frame = p.get("frame", "map")
            if frame not in ("map", "world"):
                T = self.to_map(frame, T) if self.to_map else None
                if T is None:
                    continue
            lab = det["label"]
            if lab not in best or det.get("score", 1) > best[lab][1]:
                best[lab] = (T, det.get("score", 1.0))
        self.best = best
        self.stamp = time.time() if stamp is None else stamp

    def get(self, name):
        if name not in self.best:
            return None
        T, score = self.best[name]
        return Estimate(T, self.stamp, "perception", {"score": score})

    def names(self):
        return list(self.best)


# ---------------------------------------------------------------------------------------------------- surgeon hand
class SurgeonHandSource(ABC):
    @abstractmethod
    def get(self):
        """Estimate of the surgeon's receiving hand (position is what matters) or None."""

    def holding(self, name):
        """True/False if known that the surgeon has taken `name`, None if this source cannot tell."""
        return None


class GroundTruthSurgeonHand(SurgeonHandSource):
    """CHEAT: surgeon stand-in hand site + simulated grip flag, both from /sim/objects."""

    def __init__(self, objects: GroundTruthObjects):
        self.objects = objects

    def get(self):
        return self.objects.get("surgeon_hand")

    def holding(self, name):
        e = self.objects.get(name)
        if e is None or "held_by" not in e.extra:
            return None
        return e.extra.get("held_by") == "surgeon"


class PerceptionSurgeonHand(SurgeonHandSource):
    """/g1/perception/surgeon_hand (PoseStamped, map frame after the node's TF transform). No grip knowledge."""

    def __init__(self):
        self.est = None

    def update(self, T, stamp=None):
        self.est = Estimate(T, time.time() if stamp is None else stamp, "perception")

    def get(self):
        return self.est


# ---------------------------------------------------------------------------------------------------- station
class StationEstimator(ABC):
    """Where the robot is relative to a named station (the instrument table). Returns (dx, dy, dyaw) of the station
    pose expressed in the current pelvis frame (i.e. the correction to walk), or None."""

    @abstractmethod
    def error(self, station): ...


class GroundTruthStation(StationEstimator):
    """CHEAT: pelvis ground truth (/sim/ground_truth) against the station's map pose. A table-marker estimator
    replaces it (same method, error measured from the head camera)."""

    def __init__(self, base_T_fn):
        self.base_T_fn = base_T_fn

    def error(self, station):
        T = self.base_T_fn()
        if T is None:
            return None
        yaw = math.atan2(T[1, 0], T[0, 0])
        dx, dy = station["x"] - T[0, 3], station["y"] - T[1, 3]
        c, s = math.cos(yaw), math.sin(yaw)
        return (c * dx + s * dy, -s * dx + c * dy, wrap(station["yaw"] - yaw))


def pose_yaw(T):
    from g1_mission.geometry import mat_to_quat
    return yaw_of(mat_to_quat(T[:3, :3]))
