#!/usr/bin/env python3
"""Republish the live SLAM map (/map) on a FIXED-size canvas (/map_padded) for Nav2's global costmap.

slam_toolbox grows /map as the robot explores, so map updates change its size and origin. Nav2 Humble's global
costmap resizes on each of those (StaticLayer "Resizing costmap"; it also resizes when the origin differs only by
float noise), and during the hospital survey the global_costmap node stalled within minutes of walking, right after
a resize (its clock and published footprint stopped updating, the planner kept planning from a stale start pose). Copying each /map into one fixed canvas (unknown = -1 outside the SLAM map)
means the costmap never resizes. The cell values are the SLAM map's, unchanged; nothing is added.

    map_pad_relay.py [--bounds x0,y0,x1,y1] [--in /map] [--out /map_padded]
Default bounds: the Isaac hospital + wing in the world frame (-52, -27, 35, 40), 0.05 m cells.
"""
import argparse
import array

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class Relay(Node):
    def __init__(self, a):
        super().__init__("map_pad_relay")
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in a.bounds.split(","))
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(OccupancyGrid, a.out, q)
        self.create_subscription(OccupancyGrid, a.inp, self.on_map, q)
        self.n = 0
        self.origin = None
        self.a = a
        self.last_t = None

    def on_map(self, m):
        ts = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        if self.a.min_period > 0 and self.last_t is not None and 0 <= ts - self.last_t < self.a.min_period:
            return
        self.last_t = ts
        i = m.info
        res = i.resolution
        W = int(round((self.x1 - self.x0) / res))
        H = int(round((self.y1 - self.y0) / res))
        canvas = np.full((H, W), -1, np.int8)
        src = np.asarray(m.data, np.int8).reshape(i.height, i.width)
        if self.origin is None:  # fix the canvas origin once, on the SLAM map's cell lattice
            self.origin = (i.origin.position.x - round((i.origin.position.x - self.x0) / res) * res,
                           i.origin.position.y - round((i.origin.position.y - self.y0) / res) * res)
        c0 = int(round((i.origin.position.x - self.origin[0]) / res))
        r0 = int(round((i.origin.position.y - self.origin[1]) / res))
        rs0, cs0 = max(0, -r0), max(0, -c0)
        rd0, cd0 = max(0, r0), max(0, c0)
        h = min(i.height - rs0, H - rd0)
        w = min(i.width - cs0, W - cd0)
        if h > 0 and w > 0:
            canvas[rd0:rd0 + h, cd0:cd0 + w] = src[rs0:rs0 + h, cs0:cs0 + w]
        out = OccupancyGrid()
        out.header = m.header
        out.info.map_load_time = i.map_load_time
        out.info.resolution = res
        out.info.width, out.info.height = W, H
        # identical origin and size in every message: the global costmap never resizes (a resize of the Humble
        # static layer while the planner holds the costmap is where the global_costmap node stalled)
        out.info.origin.position.x, out.info.origin.position.y = self.origin
        out.info.origin.orientation.w = 1.0
        out.data = array.array("b", canvas.tobytes())  # fast path (no per-element validation)
        self.pub.publish(out)
        self.n += 1
        if self.n % 50 == 1:
            self.get_logger().info(f"relayed {self.n} maps ({i.width}x{i.height} -> {W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounds", default="-52,-27,35,40")
    ap.add_argument("--in", dest="inp", default="/map")
    ap.add_argument("--out", default="/map_padded")
    ap.add_argument("--min-period", type=float, default=0.0,
                    help="republish at most every N s (map stamp time): fewer 2 MB DDS samples")
    a, _ = ap.parse_known_args()
    rclpy.init()
    rclpy.spin(Relay(a))


if __name__ == "__main__":
    main()
