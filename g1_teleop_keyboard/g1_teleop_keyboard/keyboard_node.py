"""Terminal keyboard teleop: raw keys (termios) -> /g1/teleop/key, one lower-case key per message
(space bar -> "space"). The teleop manager does the key mapping, velocity integration and mode logic.
Shows a live status line: mode, commanded /g1/cmd_vel, last manager message.

Run it in its own terminal (it needs a TTY):
    source env.sh && ros2 run g1_teleop_keyboard keyboard_node
Ctrl-C or Ctrl-D quits (quitting does NOT change the robot mode; press x first to damp).
"""
import json
import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from g1_interfaces.msg import Mode

NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV"}
HELP = ("r ready | l loco | w/s fwd/back | a/d strafe | q/e turn | space stop | p pause | "
        "m manip | n manip off | g nav | x E-STOP | Ctrl-C quit")


class KeyboardNode(Node):
    def __init__(self):
        super().__init__("g1_teleop_keyboard")
        self.pub = self.create_publisher(String, "/g1/teleop/key", 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode = "?"
        self.detail = ""
        self.cmd = (0.0, 0.0, 0.0)
        self.msg = ""
        self.last = ""
        self.create_subscription(Mode, "/g1/mode", self.on_mode, latched)
        self.create_subscription(Twist, "/g1/cmd_vel", self.on_cmd, 10)
        self.create_subscription(String, "/g1/teleop/manager_status", self.on_status, 10)

    def on_mode(self, m):
        self.mode = NAMES.get(m.mode, str(m.mode))
        self.detail = m.detail

    def on_cmd(self, t):
        self.cmd = (t.linear.x, t.linear.y, t.angular.z)

    def on_status(self, s):
        try:
            self.msg = json.loads(s.data).get("last_message", "")
        except ValueError:
            pass

    def send(self, key):
        self.last = key
        self.pub.publish(String(data=key))

    def line(self):
        vx, vy, wz = self.cmd
        s = (f"[{self.mode:5s}] {self.detail[:24]:24s} vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f} "
             f"| key {self.last!r:8s} | {self.msg[:60]}")
        return s


def main():
    if not sys.stdin.isatty():
        print("keyboard_node needs an interactive terminal (run it in its own terminal, not via launch)")
        return 1
    rclpy.init()
    node = KeyboardNode()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(HELP)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = os.read(fd, 1).decode(errors="ignore")
                if ch in ("\x03", "\x04"):
                    break
                if ch == "\x1b":  # swallow escape sequences (arrow keys etc.)
                    while select.select([sys.stdin], [], [], 0.01)[0]:
                        os.read(fd, 8)
                    continue
                if ch == " ":
                    node.send("space")
                elif ch.isprintable() and ch.strip():
                    node.send(ch.lower())
            cols = os.get_terminal_size().columns if sys.stdout.isatty() else 120
            sys.stdout.write("\r" + node.line()[: cols - 1].ljust(cols - 1))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
