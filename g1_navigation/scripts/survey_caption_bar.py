#!/usr/bin/env python3
"""Caption bar for the SLAM-survey screen recording: a borderless Qt strip with live sim facts.

Shows (from /clock, /sim/status and survey_runner's /survey/status): that this is a simulation, sim time, the
measured real-time factor, the current route waypoint / area, distance walked and Nav2 recoveries. It is meant to sit
under the RViz window on the recording display (no window manager), e.g.

    DISPLAY=:99 python3 survey_caption_bar.py --geometry 1920x60+0+1020

Text only reports what the topics say; nothing is estimated.
"""
import argparse
import json
import sys
import threading

import rclpy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


class Src(Node):
    def __init__(self):
        super().__init__("survey_caption_bar")
        self.clock = None
        self.sim = {}
        self.survey = {}
        self.create_subscription(Clock, "/clock", lambda m: setattr(self, "clock", m.clock.sec + 1e-9 * m.clock.nanosec), 10)
        self.create_subscription(String, "/sim/status", lambda m: self._j("sim", m), 10)
        self.create_subscription(String, "/survey/status", lambda m: self._j("survey", m), 10)

    def _j(self, k, m):
        try:
            setattr(self, k, json.loads(m.data))
        except ValueError:
            pass


def hms(s):
    s = int(max(0, s))
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}"


def geom(g):
    size, x, y = g.split("+")[0], int(g.split("+")[1]), int(g.split("+")[2])
    w, h = (int(v) for v in size.split("x"))
    return x, y, w, h


def make_panel(g, src):
    """Route progress + legend, for the empty space under the camera image."""
    from PyQt5.QtWidgets import QGridLayout, QProgressBar, QVBoxLayout
    p = QWidget()
    p.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    p.setGeometry(*geom(g))
    p.setStyleSheet("background-color: #12161e; color: #dfe5ee;")
    v = QVBoxLayout(p)
    v.setContentsMargins(22, 14, 22, 14)
    v.setSpacing(8)
    head = QLabel("Autonomous survey: Nav2 drives, lidar SLAM maps")
    head.setFont(QFont("DejaVu Sans", 15, QFont.Bold))
    head.setStyleSheet("color: #9aa4b2;")
    area = QLabel("")
    area.setFont(QFont("DejaVu Sans", 26, QFont.Bold))
    area.setStyleSheet("color: #ffffff;")
    sub = QLabel("")
    sub.setFont(QFont("DejaVu Sans", 15))
    bar = QProgressBar()
    bar.setTextVisible(False)
    bar.setFixedHeight(12)
    bar.setStyleSheet("QProgressBar {background: #262c38; border: 0px; border-radius: 6px;}"
                      "QProgressBar::chunk {background: #3ce664; border-radius: 6px;}")
    v.addWidget(head)
    v.addWidget(area)
    v.addWidget(sub)
    v.addWidget(bar)
    v.addSpacing(10)
    grid = QGridLayout()
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(6)
    items = [("#00beff", "robot trail"), ("#3ce664", "Nav2 global plan"), ("#ff28c8", "2-D lidar scan (/scan)"),
             ("rainbow", "3-D lidar, Livox MID-360 (colour = height)"), ("#ffd228", "robot (arrow)"),
             ("#33d94d", "waypoint reached"), ("#999999", "waypoint pending"), ("#f24033", "waypoint skipped"),
             ("#000000", "map: occupied"), ("#fefefe", "map: free"), ("#707e7d", "map: unknown")]
    for k, (c, t) in enumerate(items):
        sw = QLabel("")
        sw.setFixedSize(26, 14)
        if c == "rainbow":
            sw.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff0000, stop:0.33 #ffff00,"
                             " stop:0.66 #00ff00, stop:1 #0000ff); border: 1px solid #444;")
        else:
            sw.setStyleSheet(f"background: {c}; border: 1px solid #444;")
        lb = QLabel(t)
        lb.setFont(QFont("DejaVu Sans", 12))
        r, col = k % 6, (k // 6) * 2
        grid.addWidget(sw, r, col)
        grid.addWidget(lb, r, col + 1)
    v.addLayout(grid)
    v.addStretch(1)
    foot = QLabel("Simulation. Odometry in this sim is ground truth (no drift); the map is built only from the "
                  "robot's own lidar scans by slam_toolbox.")
    foot.setWordWrap(True)
    foot.setFont(QFont("DejaVu Sans", 11))
    foot.setStyleSheet("color: #8a93a0;")
    v.addWidget(foot)

    def tick():
        s = src.survey
        if not s.get("n"):
            area.setText("Starting up")
            sub.setText("waiting for Nav2 + SLAM")
            return
        i = s.get("index")
        area.setText(s.get("area") or "")
        done = s.get("reached", 0) + len(s.get("skipped", []))
        bar.setMaximum(s["n"])
        bar.setValue(done)
        txt = f"waypoint {i + 1 if i is not None else 0} of {s['n']}   ·   walked {s.get('dist_map', 0):.0f} m"
        if s.get("result"):
            txt = f"route finished: {s.get('reached', 0)} of {s['n']} waypoints reached   ·   walked {s.get('dist_map', 0):.0f} m"
        sub.setText(txt)
    p.tick = tick
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="1920x60+0+1020")
    ap.add_argument("--panel", default="", help="also show the route/legend panel at WxH+X+Y (e.g. under the camera)")
    ap.add_argument("--camera-placeholder", default="", help="WxH+X+Y: cover the RViz camera panel with a note "
                    "(live run without the camera; the view is rendered afterwards from the recorded trajectory)")
    ap.add_argument("--title", default="SIMULATION  ·  Unitree G1  ·  NVIDIA Isaac Sim  ·  lidar SLAM + Nav2")
    a, _ = ap.parse_known_args()
    rclpy.init()
    src = Src()
    threading.Thread(target=rclpy.spin, args=(src,), daemon=True).start()
    app = QApplication(sys.argv[:1])
    w = QWidget()
    w.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    size, x, y = a.geometry.split("+")[0], int(a.geometry.split("+")[1]), int(a.geometry.split("+")[2])
    W, H = (int(v) for v in size.split("x"))
    w.setGeometry(x, y, W, H)
    w.setStyleSheet("background-color: #0b0e14; color: #e8ecf2;")
    lay = QHBoxLayout(w)
    lay.setContentsMargins(18, 4, 18, 4)
    left = QLabel(a.title)
    right = QLabel("")
    f = QFont("DejaVu Sans", 15)
    left.setFont(QFont("DejaVu Sans", 15, QFont.Bold))
    right.setFont(f)
    left.setStyleSheet("color: #ffcf40;")
    right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(left)
    lay.addStretch(1)
    lay.addWidget(right)

    def tick():
        parts = []
        if src.clock is not None:
            parts.append(f"sim time {hms(src.clock)}")
        rtf = src.sim.get("rtf")
        if rtf:
            parts.append(f"sim speed {rtf:.2f}× real time")
        s = src.survey
        if s.get("n"):
            if s.get("index") is not None:
                parts.append(f"waypoint {s['index'] + 1}/{s['n']}: {s.get('area') or s.get('name')}")
            parts.append(f"walked {s.get('dist_map', 0):.0f} m")
            if s.get("recoveries"):
                parts.append(f"Nav2 recoveries {s['recoveries']}")
            if s.get("falls"):
                parts.append(f"falls {s['falls']}")
        right.setText("   ·   ".join(parts))

    panel = None
    if a.panel:
        panel = make_panel(a.panel, src)
        panel.show()

    if a.camera_placeholder:
        ph = QWidget()
        ph.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        ph.setGeometry(*geom(a.camera_placeholder))
        ph.setStyleSheet("background-color: #0e1219; color: #8a93a0;")
        lb = QLabel("Head camera view\n\nrendered afterwards in Isaac Sim from this walk's recorded\n"
                    "robot pose + joint trajectory (simulation)\n\nphysics, lidar, SLAM and Nav2 run live here")
        lb.setAlignment(Qt.AlignCenter)
        lb.setFont(QFont("DejaVu Sans", 15))
        QHBoxLayout(ph).addWidget(lb)
        ph.show()

    t = QTimer()
    t.timeout.connect(tick)
    if panel is not None:
        t.timeout.connect(panel.tick)
    t.start(250)
    w.show()
    app.exec_()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
