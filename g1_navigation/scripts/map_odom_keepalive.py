#!/usr/bin/env python3
"""Re-stamp slam_toolbox's latest map->odom at a steady rate so Nav2 always finds a current transform.

slam_toolbox (Humble 2.6, online async) only adds a scan to its graph after the robot moved 0.25 m / 0.25 rad, and the
map->odom it broadcasts carries the time of the last processed scan (+ transform_timeout). When the walking G1 stood
still or moved slowly for a while (an aborted goal, a recovery), map->odom went 10-60 s stale: Nav2's MPPI then failed
with "Transform data too old when converting from map to odom", the global costmap's robot pose froze, the robot did
not move, so SLAM never refreshed it -- a deadlock (survey 2026-10-09). This node looks up the newest map->odom and
republishes the SAME transform with the current time at 20 Hz. It never changes the estimate.

    map_odom_keepalive.py [--rate 20] [--map map] [--odom odom]   (use_sim_time is on)
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster


class KeepAlive(Node):
    def __init__(self, a):
        super().__init__("map_odom_keepalive")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.a = a
        self.br = TransformBroadcaster(self)
        self.last = None
        self.mine = set()       # stamps we published (so our own re-stamped copies are not taken as SLAM output)
        self.slam = None        # newest map->odom from slam_toolbox
        self.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        self.create_timer(1.0 / a.rate, self.tick)

    def on_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == self.a.map and t.child_frame_id == self.a.odom:
                k = (t.header.stamp.sec, t.header.stamp.nanosec)
                if k in self.mine:
                    continue
                if self.slam is None or (k >= (self.slam.header.stamp.sec, self.slam.header.stamp.nanosec)):
                    self.slam = t

    def tick(self):
        if self.slam is None:
            return
        t = self.slam
        now = self.get_clock().now()
        if self.last is not None and now.nanoseconds <= self.last:
            return
        self.last = now.nanoseconds
        from geometry_msgs.msg import TransformStamped
        out = TransformStamped()
        out.header.frame_id, out.child_frame_id = t.header.frame_id, t.child_frame_id
        out.transform = t.transform
        out.header.stamp = now.to_msg()
        self.mine.add((out.header.stamp.sec, out.header.stamp.nanosec))
        if len(self.mine) > 2000:
            self.mine = set(sorted(self.mine)[-1000:])
        self.br.sendTransform(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--map", default="map")
    ap.add_argument("--odom", default="odom")
    a, _ = ap.parse_known_args()
    rclpy.init()
    rclpy.spin(KeepAlive(a))


if __name__ == "__main__":
    main()
