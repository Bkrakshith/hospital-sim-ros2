"""Record skill executions as a LeRobot dataset (format v2.1: parquet per episode + MP4 per camera + meta/*).
Off by default (mission param `record_dir`). One episode = one skill execution (pick, handover, ...), task string =
"<skill> <instrument>" (e.g. "pick scalpel_handle").

State (29-D, `observation.state`), GR00T-style groups:
    waist      [0:3]    waist_yaw, waist_roll, waist_pitch                       (rad, measured)
    left_arm   [3:10]   shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw     (rad, measured)
    right_arm  [10:17]  same order                                              (rad, measured)
    left_hand  [17:23]  thumb, thumb_aux, index, middle, ring, pinky            (0 open .. 1 closed, measured)
    right_hand [23:29]  same order
Action (32-D, `action`) = exactly what the skill commanded on the /g1/auto/* channels:
    waist [0:3], left_arm [3:10], right_arm [10:17] (joint targets = /g1/auto/upper_body_target),
    left_hand [17:23], right_hand [23:29] (/g1/auto/hand/*/cmd), base_velocity [29:32] (vx, vy, wz = cmd_vel)
Images: `observation.images.head` = /g1/head_camera/image_raw (RGB 640x360, as the sim and the Isaac head camera
publish it; other sizes are resized to `image_size`).
No base/pelvis state is in the 29-D vector (arm skills run standing; add `observation.base_velocity` if a
walking skill is ever learned). Frames are sampled at `fps` from the latest state/action (zero-order hold).
"""
import json
import os
import time

import numpy as np

UPPER = ["waist_yaw", "waist_roll", "waist_pitch",
         "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
         "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
         "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
         "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
HAND = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
STATE_NAMES = UPPER + [f"left_hand_{h}" for h in HAND] + [f"right_hand_{h}" for h in HAND]
ACTION_NAMES = STATE_NAMES + ["base_vx", "base_vy", "base_wz"]
GROUPS = {"waist": [0, 3], "left_arm": [3, 10], "right_arm": [10, 17], "left_hand": [17, 23], "right_hand": [23, 29]}
ACTION_GROUPS = dict(GROUPS, base_velocity=[29, 32])


def state_vector(robot):
    q = robot.upper_q()
    hl, hr = robot.hand_q("left"), robot.hand_q("right")
    if q is None or hl is None or hr is None:
        return None
    return np.concatenate([q, hl, hr]).astype(np.float32)


def action_vector(upper, left_hand, right_hand, cmd_vel):
    return np.concatenate([upper, left_hand, right_hand, cmd_vel]).astype(np.float32)


def split_action(a):
    a = np.asarray(a, float)
    return {"upper": a[0:17], "left_hand": a[17:23], "right_hand": a[23:29], "cmd_vel": tuple(a[29:32])}


class LeRobotRecorder:
    def __init__(self, root, fps=15, robot_type="unitree_g1_29dof_brainco", image_size=(360, 640)):
        self.root, self.fps, self.robot_type = root, fps, robot_type
        self.h, self.w = image_size
        os.makedirs(os.path.join(root, "meta"), exist_ok=True)
        self.info_path = os.path.join(root, "meta", "info.json")
        self.tasks = self._read_jsonl("tasks.jsonl")
        self.episodes = self._read_jsonl("episodes.jsonl")
        self.ep = None
        self.last_action = np.zeros(32, np.float32)
        self.last_action_set = False

    def _read_jsonl(self, name):
        p = os.path.join(self.root, "meta", name)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write_jsonl(self, name, rows):
        with open(os.path.join(self.root, "meta", name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    @property
    def recording(self):
        return self.ep is not None

    # ------------------------------------------------------------------ episode
    def start(self, task):
        idx = len(self.episodes)
        tid = next((t["task_index"] for t in self.tasks if t["task"] == task), None)
        if tid is None:
            tid = len(self.tasks)
            self.tasks.append({"task_index": tid, "task": task})
        self.ep = {"index": idx, "task": task, "task_index": tid, "state": [], "action": [], "t": [], "t0": None,
                   "writer": None, "video": None, "next_t": 0.0}
        self.last_action_set = False

    def action(self, upper=None, left_hand=None, right_hand=None, cmd_vel=None):
        """Called by skills with whatever they commanded this tick (partial updates allowed)."""
        a = self.last_action
        if upper is not None:
            a[0:17] = upper
        if left_hand is not None:
            a[17:23] = left_hand
        if right_hand is not None:
            a[23:29] = right_hand
        if cmd_vel is not None:
            a[29:32] = cmd_vel
        self.last_action_set = True

    def frame(self, robot, now=None, image=None):
        """Sample one frame if `1/fps` has passed. `image`: RGB uint8 (H, W, 3) or None (black frame)."""
        if self.ep is None:
            return
        ep = self.ep
        now = robot.now() if now is None else now
        if ep["t0"] is None:
            ep["t0"] = now
        t = now - ep["t0"]
        if t + 1e-6 < ep["next_t"]:
            return
        st = state_vector(robot)
        if st is None:
            return
        if not self.last_action_set:          # before the first command: action = hold the current state
            self.last_action[:29] = st
            self.last_action[29:] = 0.0
        ep["next_t"] = (len(ep["t"]) + 1) / self.fps
        ep["t"].append(len(ep["t"]) / self.fps)
        ep["state"].append(st.copy())
        ep["action"].append(self.last_action.copy())
        self._video(image)

    def _video(self, image):
        ep = self.ep
        if ep["writer"] is None:
            import imageio.v2 as iio
            chunk = ep["index"] // 1000
            d = os.path.join(self.root, "videos", f"chunk-{chunk:03d}", "observation.images.head")
            os.makedirs(d, exist_ok=True)
            ep["video"] = os.path.join(d, f"episode_{ep['index']:06d}.mp4")
            ep["writer"] = iio.get_writer(ep["video"], fps=self.fps, codec="libx264", quality=None,
                                          ffmpeg_params=["-crf", "23", "-pix_fmt", "yuv420p", "-g", "2"],
                                          macro_block_size=1)
        if image is None:
            img = np.zeros((self.h, self.w, 3), np.uint8)
        else:
            img = image
            if img.shape[0] != self.h or img.shape[1] != self.w:
                import cv2
                img = cv2.resize(img, (self.w, self.h))
        ep["writer"].append_data(img)

    def stop(self, success, discard=False):
        """Close the episode. Returns the episode index, or None if nothing was written."""
        ep, self.ep = self.ep, None
        if ep is None:
            return None
        if ep["writer"] is not None:
            ep["writer"].close()
        n = len(ep["t"])
        if discard or n == 0:
            if ep["video"] and os.path.exists(ep["video"]):
                os.remove(ep["video"])
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq
        total = sum(e["length"] for e in self.episodes)
        chunk = ep["index"] // 1000
        d = os.path.join(self.root, "data", f"chunk-{chunk:03d}")
        os.makedirs(d, exist_ok=True)
        S = np.stack(ep["state"])
        A = np.stack(ep["action"])
        table = pa.table({
            "observation.state": pa.array(list(S), type=pa.list_(pa.float32(), 29)),
            "action": pa.array(list(A), type=pa.list_(pa.float32(), 32)),
            "timestamp": pa.array(np.array(ep["t"], np.float32)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep["index"], np.int64)),
            "index": pa.array(np.arange(total, total + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, ep["task_index"], np.int64)),
            "next.success": pa.array(np.r_[np.zeros(n - 1, bool), [bool(success)]]),
        })
        pq.write_table(table, os.path.join(d, f"episode_{ep['index']:06d}.parquet"))
        self.episodes.append({"episode_index": ep["index"], "tasks": [ep["task"]], "length": n,
                              "success": bool(success)})
        self._write_jsonl("episodes.jsonl", self.episodes)
        self._write_jsonl("tasks.jsonl", self.tasks)
        stats = self._read_jsonl("episodes_stats.jsonl")

        def st(x):
            return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
                    "std": x.std(0).tolist(), "count": [len(x)]}
        stats.append({"episode_index": ep["index"], "stats": {"observation.state": st(S), "action": st(A)}})
        self._write_jsonl("episodes_stats.jsonl", stats)
        self._write_info()
        return ep["index"]

    def _write_info(self):
        n_ep = len(self.episodes)
        info = {
            "codebase_version": "v2.1", "robot_type": self.robot_type, "fps": self.fps,
            "total_episodes": n_ep, "total_frames": int(sum(e["length"] for e in self.episodes)),
            "total_tasks": len(self.tasks), "total_videos": n_ep, "total_chunks": n_ep // 1000 + 1,
            "chunks_size": 1000, "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.head": {
                    "dtype": "video", "shape": [self.h, self.w, 3], "names": ["height", "width", "channel"],
                    "info": {"video.fps": self.fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "has_audio": False}},
                "observation.state": {"dtype": "float32", "shape": [29], "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": [32], "names": ACTION_NAMES},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "next.success": {"dtype": "bool", "shape": [1], "names": None},
            },
            "g1_modality": {"state": GROUPS, "action": ACTION_GROUPS,
                            "note": "GR00T-style groups; see g1_mission/recorder.py"},
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=1)
