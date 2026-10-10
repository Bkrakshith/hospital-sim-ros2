#!/usr/bin/env python3
"""Walk the G1 through a survey route with Nav2 while slam_toolbox maps (config/survey_route.yaml).

The robot drives itself: each waypoint is sent to Nav2 as a NavigateToPose goal on the LIVE map (Nav2 + SLAM,
`nav.launch.py slam:=true`), in mode NAV (key `g` after `r`). This node only sequences the goals:

  * route waypoints are in the `map` frame; with the default start (sim spawn at the world origin, yaw 0) the SLAM
    `map` frame is the world frame, so the route, docs/interfaces.md locations and speed zones all line up;
  * goal yaw = heading towards the next waypoint (no turn on the spot after arriving); `yaw:` in the route overrides;
  * a goal that Nav2 aborts (or that times out) is retried after clearing both costmaps, `--retries` times, then
    skipped; skipped waypoints are logged and reported;
  * mode watchdog: if the robot is no longer in NAV it re-requests NAV (`g`); after a fall (DAMP, source safety)
    it can re-lift the robot with the sim harness (`--fall-recovery harness`, a SIM-ONLY intervention that is
    logged as such) or stop (default `stop`).

Ground truth (/sim/ground_truth) is only LOGGED (distance walked, distance to the stairs), never used to steer.

Published for RViz / video: /survey/trail (nav_msgs/Path, map->pelvis from TF), /survey/robot_pose (PoseStamped,
2 Hz, a big arrow for the whole-building view), /survey/route (MarkerArray:
grey pending, yellow current, green reached, red skipped), /survey/status (std_msgs/String JSON, 1 Hz).
Log: <log-dir>/survey_events.jsonl (every event with wall + sim time) and survey_summary.json.

    ros2 run g1_navigation survey_runner.py --route <yaml> --log-dir <dir> [--start-at NAME] [--only N]
"""
import argparse
import json
import math
import os
import signal
import subprocess
import time

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from g1_interfaces.msg import Mode
from geometry_msgs.msg import Point, PolygonStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

MODE_NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV", 5: "AUTO", 6: "HOLD"}


def yaw_q(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def rect_dist(x, y, r):
    x0, y0, x1, y1 = r
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


class Survey(Node):
    def __init__(self, a):
        super().__init__("survey_runner")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.a = a
        self.route = yaml.safe_load(open(a.route))
        self.wps = self.route["waypoints"]
        self.stairs = [tuple(s["box"]) for s in self.route.get("stairs", [])]
        os.makedirs(a.log_dir, exist_ok=True)
        evp = os.path.join(a.log_dir, "survey_events.jsonl")
        self.prev_events = [json.loads(ln) for ln in open(evp)] if os.path.exists(evp) else []
        self.ev = open(evp, "a")
        self.trail_csv = open(os.path.join(a.log_dir, "trail_map.csv"), "a")
        self.tf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.key_pub = self.create_publisher(String, "/g1/teleop/key", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.trail_pub = self.create_publisher(Path, "/survey/trail", latched)
        self.route_pub = self.create_publisher(MarkerArray, "/survey/route", latched)
        self.status_pub = self.create_publisher(String, "/survey/status", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/survey/robot_pose", 10)
        self.create_subscription(Mode, "/g1/mode", self.on_mode, latched)
        self.create_subscription(Odometry, "/sim/ground_truth", self.on_gt, 10)
        self.clear_srvs = [self.create_client(Empty, n) for n in (
            "/global_costmap/clear_entirely_global_costmap", "/local_costmap/clear_entirely_local_costmap")]
        self.harness_srv = self.create_client(SetBool, "/sim/harness")
        self.fp = {"global": None, "local": None}
        for k in self.fp:
            self.create_subscription(PolygonStamped, f"/{k}_costmap/published_footprint",
                                     lambda m, k=k: self.fp.__setitem__(k, m.header.stamp.sec + 1e-9 *
                                                                        m.header.stamp.nanosec), 10)
        self.nav_proc = None
        self.nav_restarts = 0
        self.scan = None
        self.escapes = 0
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.vel_pub = self.create_publisher(Twist, "/g1/cmd_vel_nav", 10)
        self.mode = None
        self.mode_src = ""
        self.gt = None
        self.gt_prev = None
        self.gt_dist = 0.0
        self.min_stairs = 1e9
        self.trail = Path()
        self.trail.header.frame_id = "map"
        self.map_dist = 0.0
        self.last_map_xy = None
        self.state = {i: "pending" for i in range(len(self.wps))}
        self.cur = None
        self.recoveries = 0
        self.falls = 0
        self.fb = {}
        self.result = None
        self.t_start_sim = None
        self.t_start_wall = time.time()
        self.resume_state()
        self.create_timer(0.5, self.on_trail_timer)
        self.create_timer(1.0, self.publish_status)

    def resume_state(self):
        """after a runner restart (same --log-dir): keep reached/skipped marks, counters and the drawn trail."""
        for e in self.prev_events:
            if e.get("event") == "goal_result":
                if e["outcome"] == "succeeded":
                    self.state[e["index"]] = "reached"
                self.recoveries += int(e.get("recoveries", 0))
            elif e.get("event") == "waypoint_skipped":
                self.state[e["index"]] = "skipped"
            elif e.get("event") == "nav2_restart":
                self.nav_restarts = max(self.nav_restarts, int(e.get("n", 0)))
            elif e.get("event") == "fall":
                self.falls += 1
        tp = os.path.join(self.a.log_dir, "trail_map.csv")
        if os.path.exists(tp):
            for ln in open(tp):
                try:
                    x, y = (float(v) for v in ln.split(",")[1:3])
                except ValueError:
                    continue
                ps = PoseStamped()
                ps.header.frame_id = "map"
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, 0.02
                ps.pose.orientation.w = 1.0
                if self.last_map_xy is not None:
                    self.map_dist += math.hypot(x - self.last_map_xy[0], y - self.last_map_xy[1])
                self.last_map_xy = (x, y)
                self.trail.poses.append(ps)
        if self.prev_events:
            self.get_logger().info(f"resumed: {sum(v == 'reached' for v in self.state.values())} reached, "
                                   f"{len(self.trail.poses)} trail points, {self.map_dist:.0f} m")

    # ------------------------------------------------------------------ helpers
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def log(self, kind, **kw):
        rec = {"wall": round(time.time(), 3), "sim": round(self.now_s(), 3), "event": kind, **kw}
        self.ev.write(json.dumps(rec) + "\n")
        self.ev.flush()
        self.get_logger().info(f"{kind} {json.dumps(kw)}")

    def spin_for(self, sec_wall):
        end = time.time() + sec_wall
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def spin_until(self, cond, timeout_sim, poll=0.05):
        t0 = self.now_s()
        w0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=poll)
            if cond():
                return True
            if self.now_s() - t0 > timeout_sim and time.time() - w0 > 2.0:
                return False
        return False

    def key(self, k):
        self.key_pub.publish(String(data=k))
        self.log("key", key=k)

    def on_mode(self, m):
        prev = self.mode
        self.mode, self.mode_src = m.mode, m.source
        if prev != m.mode:
            self.log("mode", mode=MODE_NAMES.get(m.mode, m.mode), source=m.source, detail=m.detail)

    def on_gt(self, m):
        p = m.pose.pose.position
        self.gt = (p.x, p.y, p.z)
        if self.gt_prev is not None and self.t_start_sim is not None:
            self.gt_dist += math.hypot(p.x - self.gt_prev[0], p.y - self.gt_prev[1])
        self.gt_prev = self.gt
        for r in self.stairs:
            self.min_stairs = min(self.min_stairs, rect_dist(p.x, p.y, r))

    def robot_xy_yaw(self):
        try:
            t = self.tf.lookup_transform("map", "pelvis", rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw, t

    def on_trail_timer(self):
        r = self.robot_xy_yaw()
        if r is None:
            return
        x, y, yaw, t = r
        rp = PoseStamped()
        rp.header.frame_id = "map"
        rp.header.stamp = t.header.stamp
        rp.pose.position.x, rp.pose.position.y, rp.pose.position.z = x, y, 0.1
        rp.pose.orientation = t.transform.rotation
        self.pose_pub.publish(rp)
        if self.t_start_sim is None:
            return
        if self.last_map_xy is not None:
            d = math.hypot(x - self.last_map_xy[0], y - self.last_map_xy[1])
            if d < 0.05:
                return
            self.map_dist += d
        self.last_map_xy = (x, y)
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = t.header.stamp
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, 0.02
        ps.pose.orientation = t.transform.rotation
        self.trail.poses.append(ps)
        self.trail_csv.write(f"{self.now_s():.2f},{x:.3f},{y:.3f}\n")
        self.trail_csv.flush()
        self.trail.header.stamp = t.header.stamp
        self.trail_pub.publish(self.trail)

    def publish_route(self):
        ma = MarkerArray()
        colours = {"pending": (0.6, 0.6, 0.6), "current": (1.0, 0.85, 0.1), "reached": (0.2, 0.85, 0.3),
                   "skipped": (0.95, 0.25, 0.2)}
        line = Marker()
        line.header.frame_id = "map"
        line.ns, line.id, line.type, line.action = "route_line", 0, Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.05
        line.color.r, line.color.g, line.color.b, line.color.a = 0.55, 0.55, 0.6, 0.6
        line.pose.orientation.w = 1.0
        st = self.route["start"]
        line.points.append(Point(x=float(st["x"]), y=float(st["y"]), z=0.03))
        for i, w in enumerate(self.wps):
            m = Marker()
            m.header.frame_id = "map"
            m.ns, m.id, m.type, m.action = "waypoints", i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(w["x"]), float(w["y"]), 0.05
            m.pose.orientation.w = 1.0
            s = 0.45 if self.state[i] == "current" else 0.3
            m.scale.x = m.scale.y = m.scale.z = s
            c = colours[self.state[i]]
            m.color.r, m.color.g, m.color.b, m.color.a = c[0], c[1], c[2], 0.9
            ma.markers.append(m)
            line.points.append(Point(x=float(w["x"]), y=float(w["y"]), z=0.03))
        ma.markers.append(line)
        self.route_pub.publish(ma)

    def status(self):
        i = self.cur
        w = self.wps[i] if i is not None else None
        el = self.now_s() - self.t_start_sim if self.t_start_sim is not None else 0.0
        return {"index": i, "n": len(self.wps), "name": w["name"] if w else None, "area": w.get("area") if w else None,
                "reached": sum(1 for v in self.state.values() if v == "reached"),
                "skipped": [self.wps[k]["name"] for k, v in self.state.items() if v == "skipped"],
                "sim_elapsed": round(el, 1), "wall_elapsed": round(time.time() - self.t_start_wall, 1),
                "dist_map": round(self.map_dist, 1), "dist_gt": round(self.gt_dist, 1),
                "recoveries": self.recoveries, "falls": self.falls, "nav2_restarts": self.nav_restarts,
                "escapes": self.escapes, "mode": MODE_NAMES.get(self.mode, self.mode),
                "min_stairs_gt": round(self.min_stairs, 2) if self.min_stairs < 1e8 else None,
                "nav_feedback": self.fb, "result": self.result}

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.status())))

    # ------------------------------------------------------------------ mode handling
    def ensure_nav(self):
        if self.mode == Mode.NAV:
            return True
        if self.mode == Mode.DAMP and self.mode_src == "safety":
            self.falls += 1
            self.log("fall", at=self.gt, mode_source=self.mode_src)
            if self.a.fall_recovery != "harness":
                return False
            # SIM-ONLY intervention: re-lift with the virtual harness at the current x, y
            self.harness_srv.wait_for_service(timeout_sec=5.0)
            self.harness_srv.call_async(SetBool.Request(data=True))
            self.log("sim_intervention", what="harness re-lift after fall")
            self.spin_until(lambda: False, 3.0)
        if self.mode == Mode.DAMP:
            self.key("r")
            if not self.spin_until(lambda: self.mode == Mode.STAND, 15.0):
                return False
            self.spin_until(lambda: False, 6.0)  # stand-up (2 s) + harness release (2.5 s + 1 s fade)
        if self.mode == Mode.HOLD:
            self.key("r")
            self.spin_until(lambda: self.mode == Mode.STAND, 10.0)
        if self.mode == Mode.STAND:
            self.key("g")
            return self.spin_until(lambda: self.mode == Mode.NAV, 10.0)
        return self.mode == Mode.NAV

    def clear_costmaps(self):
        for c in self.clear_srvs:
            if c.service_is_ready():
                c.call_async(Empty.Request())
        self.spin_for(0.5)

    # ------------------------------------------------------------------ main
    def goal_yaw(self, i):
        w = self.wps[i]
        if w.get("yaw") is not None:
            return float(w["yaw"])
        nxt = self.wps[i + 1] if i + 1 < len(self.wps) else None
        if nxt is not None:
            return math.atan2(nxt["y"] - w["y"], nxt["x"] - w["x"])
        prev = self.wps[i - 1] if i > 0 else self.route["start"]
        return math.atan2(w["y"] - prev["y"], w["x"] - prev["x"])

    def run_goal(self, i):
        w = self.wps[i]
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(w["x"]), float(w["y"])
        qz, qw = yaw_q(self.goal_yaw(i))
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        goal = NavigateToPose.Goal()
        goal.pose = ps
        r = self.robot_xy_yaw()
        d0 = math.hypot(w["x"] - r[0], w["y"] - r[1]) if r else 10.0
        timeout = max(self.a.min_goal_time, 6.0 * d0 / 0.3)  # sim seconds
        fut = self.nav.send_goal_async(goal, feedback_callback=self.on_fb)
        if not self.spin_until(lambda: fut.done(), 30.0):
            return "send_timeout", 0
        gh = fut.result()
        if not gh.accepted:
            self.spin_until(lambda: False, 3.0)  # bt_navigator not active yet (e.g. right after a Nav2 restart)
            return "rejected", 0
        res = gh.get_result_async()
        t0 = self.now_s()
        while rclpy.ok() and not res.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.mode != Mode.NAV:
                self.log("mode_lost_during_goal", name=w["name"], mode=MODE_NAMES.get(self.mode, self.mode))
                gh.cancel_goal_async()
                self.spin_for(1.0)
                return "mode_lost", self.fb.get("recoveries", 0)
            if self.now_s() - t0 > timeout:
                gh.cancel_goal_async()
                self.spin_for(1.0)
                return "timeout", self.fb.get("recoveries", 0)
            stalled = self.nav_stalled()
            if stalled:
                self.log("nav2_stalled", name=w["name"], lag=stalled)
                gh.cancel_goal_async()
                self.spin_for(1.0)
                return "nav2_stalled", self.fb.get("recoveries", 0)
        st = res.result().status
        recs = self.fb.get("recoveries", 0)
        return {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
                GoalStatus.STATUS_CANCELED: "canceled"}.get(st, f"status_{st}"), recs

    def nav_stalled(self):
        """costmap footprints are published every update with the costmap node's clock; if one lags the sim clock
        by more than --stall-s, that costmap has stopped updating (seen in planner_server with Isaac)."""
        if self.nav_proc is None or self.a.stall_s <= 0:
            return None
        now = self.now_s()
        for k, t in self.fp.items():
            if t is not None and now - t > self.a.stall_s:
                return {k: round(now - t, 1)}
        return None

    def start_nav(self):
        logf = open(self.a.nav_log, "a") if self.a.nav_log else subprocess.DEVNULL
        for tries in range(4):
            self.nav_proc = subprocess.Popen(self.a.nav_cmd, shell=True, start_new_session=True, stdout=logf,
                                             stderr=subprocess.STDOUT)
            time.sleep(8.0)
            if self.nav_proc.poll() is None:
                break
            self.log("nav2_launch_exited", code=self.nav_proc.returncode, attempt=tries)
            time.sleep(5.0)
        self.fp = {"global": None, "local": None}
        self.log("nav2_started", pid=self.nav_proc.pid)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.nav.wait_for_server()
        # wait until both costmaps publish fresh footprints
        self.spin_until(lambda: all(t is not None and self.now_s() - t < 3.0 for t in self.fp.values()), 60.0)

    def restart_nav(self):
        self.nav_restarts += 1
        before = self.robot_xy_yaw()  # AMCL's estimate (map frame) before the restart, not ground truth
        self.log("nav2_restart", n=self.nav_restarts)
        if self.nav_proc == "adopted":  # started by a previous runner: kill by name (never ourselves)
            pats = ["ros2 launch g1_navigation nav.launch.py",
                    "controller_server|planner_server|bt_navigator|waypoint_follower|velocity_smoother|"
                    "behavior_server|smoother_server|lifecycle_manager_navigation|lifecycle_manager_localization|"
                    "lifecycle_manager_filters|nav2_amcl/amcl|nav2_map_server/map_server|costmap_filter_info_server"]
            for pat, sig, wait in ((pats[0], signal.SIGINT, 8.0), (pats[1], signal.SIGKILL, 2.0)):
                out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
                for pid in out:
                    if int(pid) not in (os.getpid(), os.getppid()):
                        try:
                            os.kill(int(pid), sig)
                        except ProcessLookupError:
                            pass
                time.sleep(wait)
        elif self.nav_proc is not None:
            for sig, wait in ((signal.SIGINT, 10), (signal.SIGKILL, 3)):
                try:
                    os.killpg(self.nav_proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.nav_proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
        self.spin_for(3.0)
        self.start_nav()
        if (self.a.initial_pose or self.a.initial_pose_on_restart) and before is not None:  # AMCL restarted too: hand it its own last estimate
            ip = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
            for _ in range(3):
                m = PoseWithCovarianceStamped()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.pose.pose.position.x, m.pose.pose.position.y = before[0], before[1]
                m.pose.pose.orientation.z, m.pose.pose.orientation.w = yaw_q(before[2])
                m.pose.covariance[0] = m.pose.covariance[7] = 0.5 ** 2
                m.pose.covariance[35] = 0.2 ** 2
                ip.publish(m)
                self.spin_for(1.0)
            self.log("initial_pose_after_restart", x=round(before[0], 2), y=round(before[1], 2), source="amcl_last")

    def escape(self, name):
        """scripted un-wedging (the robot got pushed against an obstacle and Nav2 can neither plan nor back up):
        step 0.6-0.8 m towards the most open direction of the latest 2-D scan, in NAV mode, then let Nav2 retry."""
        m = self.scan
        if m is None or self.mode != Mode.NAV:
            return
        r = np.asarray(m.ranges, float)
        r = np.where(np.isfinite(r), r, m.range_max)
        k = 30  # +-15 deg window
        sm = np.array([r[np.arange(i - k, i + k) % len(r)].min() for i in range(len(r))])
        i = int(np.argmax(sm))
        a = m.angle_min + i * m.angle_increment
        self.escapes += 1
        self.log("escape_maneuver", name=name, direction_deg=round(math.degrees(a), 1), clearance=round(sm[i], 2))
        # turn to face the open direction (the walking policy is poor at side-stepping), then walk ~0.8 m
        r0 = self.robot_xy_yaw()
        if r0 is None:
            return
        target = r0[2] + a
        t0 = self.now_s()
        while rclpy.ok() and self.now_s() - t0 < 8.0:
            r = self.robot_xy_yaw()
            err = math.atan2(math.sin(target - r[2]), math.cos(target - r[2]))
            if abs(err) < 0.25:
                break
            tw = Twist()
            tw.angular.z = 0.5 if err > 0 else -0.5
            self.vel_pub.publish(tw)
            self.spin_for(0.1)
        t0 = self.now_s()
        while rclpy.ok() and self.now_s() - t0 < 3.0:
            tw = Twist()
            tw.linear.x = 0.3
            self.vel_pub.publish(tw)
            self.spin_for(0.1)
        self.vel_pub.publish(Twist())
        self.spin_until(lambda: False, 1.0)

    def on_fb(self, msg):
        f = msg.feedback
        self.fb = {"dist_remaining": round(f.distance_remaining, 2), "recoveries": int(f.number_of_recoveries),
                   "nav_time": round(f.navigation_time.sec + 1e-9 * f.navigation_time.nanosec, 1)}

    def run(self):
        a = self.a
        self.log("start", route=a.route, waypoints=len(self.wps))
        self.publish_route()
        if a.initial_pose:  # AMCL on a saved map: the operator's initial pose (like the dashboard's "Initial pose")
            ip = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
            x, y, yaw = (float(v) for v in a.initial_pose.split(","))
            self.spin_until(lambda: self.now_s() > 0, 1e9)
            if a.nav_cmd:
                self.start_nav()
                a.nav_cmd = ""
            for _ in range(3):
                m = PoseWithCovarianceStamped()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.pose.pose.position.x, m.pose.pose.position.y = x, y
                m.pose.pose.orientation.z, m.pose.pose.orientation.w = yaw_q(yaw)
                m.pose.covariance[0] = m.pose.covariance[7] = 0.5 ** 2
                m.pose.covariance[35] = 0.15 ** 2
                ip.publish(m)
                self.spin_for(1.0)
            self.log("initial_pose", x=x, y=y, yaw=yaw)
        self.get_logger().info("waiting for /clock, TF map->pelvis and Nav2 ...")
        self.spin_until(lambda: self.now_s() > 0 and self.robot_xy_yaw() is not None, 1e9)
        if a.nav_cmd and not a.adopt_nav:
            self.start_nav()
        elif a.nav_cmd:
            self.nav_proc = "adopted"
        self.nav.wait_for_server()
        self.spin_until(lambda: self.mode is not None, 10.0)
        if not self.ensure_nav():
            self.log("abort", reason="could not enter NAV")
            return
        self.t_start_sim = self.now_s()
        first = 0
        if a.start_at:
            first = next(i for i, w in enumerate(self.wps) if w["name"] == a.start_at)
        last = len(self.wps) if a.only <= 0 else min(len(self.wps), first + a.only)
        r = self.robot_xy_yaw()
        self.log("walk_start", map_xy=[round(r[0], 2), round(r[1], 2)] if r else None, gt=self.gt)
        order = list(range(first, last))
        k_ = 0
        second = False
        while k_ < len(order):
            i = order[k_]
            k_ += 1
            if (a.second_pass and not second and k_ == len(order) and last == len(self.wps)):
                # before the final waypoint (back to the start): one more try at every skipped waypoint, nearest first
                second = True
                todo = [j for j in range(len(self.wps) - 1) if self.state[j] == "skipped"]
                redo, (px, py) = [], (self.robot_xy_yaw() or (0.0, 0.0))[:2]
                while todo:  # greedy nearest-first from where the robot is
                    j = min(todo, key=lambda j: math.hypot(self.wps[j]["x"] - px, self.wps[j]["y"] - py))
                    todo.remove(j)
                    redo.append(j)
                    px, py = self.wps[j]["x"], self.wps[j]["y"]
                if redo:
                    self.log("second_pass", waypoints=[self.wps[j]["name"] for j in redo])
                    order[k_ - 1:k_ - 1] = redo
                    i = order[k_ - 1]
            w = self.wps[i]
            self.cur = i
            self.state[i] = "current"
            self.publish_route()
            outcome = None
            attempt = 0
            restarts_here = 0
            rejects_here = 0
            while attempt <= a.retries:
                if not self.ensure_nav():
                    self.log("abort", reason="robot not in NAV (fall?)", at=w["name"])
                    self.finish("aborted")
                    return
                ts = self.now_s()
                self.fb = {}
                self.log("goal", index=i, name=w["name"], area=w.get("area"), x=w["x"], y=w["y"], attempt=attempt)
                outcome, recs = self.run_goal(i)
                self.recoveries += recs
                r = self.robot_xy_yaw()
                self.log("goal_result", index=i, name=w["name"], outcome=outcome, attempt=attempt,
                         sim_s=round(self.now_s() - ts, 1), recoveries=recs,
                         map_xy=[round(r[0], 2), round(r[1], 2)] if r else None,
                         gt_xy=[round(self.gt[0], 2), round(self.gt[1], 2)] if self.gt else None)
                if outcome == "succeeded":
                    break
                if outcome == "rejected" and rejects_here < 20:
                    rejects_here += 1
                    continue  # Nav2 still activating: does not use up an attempt
                if outcome == "nav2_stalled" and self.nav_proc is not None and restarts_here < 5:
                    restarts_here += 1
                    self.restart_nav()
                    continue  # a Nav2 restart does not use up an attempt
                attempt += 1
                if outcome == "aborted":
                    self.escape(w["name"])
                self.clear_costmaps()
            self.state[i] = "reached" if outcome == "succeeded" else "skipped"
            if outcome != "succeeded":
                self.log("waypoint_skipped", index=i, name=w["name"], last_outcome=outcome)
            self.publish_route()
        self.finish("done")

    def finish(self, how):
        self.result = how
        if self.nav_proc is not None and not isinstance(self.nav_proc, str) and self.a.stop_nav_at_end:
            try:
                os.killpg(self.nav_proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        s = self.status()
        s["result"] = how
        s["waypoints"] = [{"name": w["name"], "area": w.get("area"), "state": self.state[i]}
                          for i, w in enumerate(self.wps)]
        self.log("finish", result=how, reached=s["reached"], skipped=s["skipped"], dist_map=s["dist_map"],
                 dist_gt=s["dist_gt"], sim_elapsed=s["sim_elapsed"], recoveries=s["recoveries"], falls=s["falls"],
                 min_stairs_gt=s["min_stairs_gt"])
        json.dump(s, open(os.path.join(self.a.log_dir, "survey_summary.json"), "w"), indent=1)
        self.publish_status()
        if self.a.stand_at_end and self.mode == Mode.NAV:
            self.key("p")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--route", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--start-at", default="", help="waypoint name to start from (resume)")
    ap.add_argument("--only", type=int, default=0, help="only this many waypoints (rehearsal)")
    ap.add_argument("--retries", type=int, default=2, help="retries per waypoint after the first attempt")
    ap.add_argument("--min-goal-time", type=float, default=150.0, help="per-goal timeout floor, sim s")
    ap.add_argument("--fall-recovery", choices=["stop", "harness"], default="stop")
    ap.add_argument("--stand-at-end", action="store_true", help="key p (NAV -> STAND) after the last waypoint")
    ap.add_argument("--nav-cmd", default="", help="shell command that launches Nav2; the runner then owns it and "
                    "restarts it when a costmap stops updating (watchdog)")
    ap.add_argument("--nav-log", default="", help="append the Nav2 launch output here")
    ap.add_argument("--stall-s", type=float, default=20.0, help="watchdog: costmap footprint older than this (sim s)")
    ap.add_argument("--stop-nav-at-end", action="store_true")
    ap.add_argument("--initial-pose-on-restart", action="store_true",
                    help="after a Nav2 restart, hand AMCL its own last estimate (also without --initial-pose)")
    ap.add_argument("--second-pass", action="store_true", help="retry skipped waypoints once before the last one")
    ap.add_argument("--initial-pose", default="", help="x,y,yaw: publish /initialpose (AMCL) before starting")
    ap.add_argument("--adopt-nav", action="store_true", help="with --nav-cmd: a Nav2 is already running (runner "
                    "restart); do not launch another one until the watchdog needs a restart")
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Survey(a)
    try:
        n.run()
        n.spin_for(3.0)
    except KeyboardInterrupt:
        n.finish("interrupted")
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
