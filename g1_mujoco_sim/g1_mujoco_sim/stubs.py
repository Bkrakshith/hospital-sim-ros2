"""Robot-side stubs the sim provides so the UI safety chain can be exercised without hardware (docs/interfaces.md 3.4):

  * /g1/estop/heartbeat (std_msgs/Bool, 10 Hz): the hardware wireless e-stop receiver. True = released (healthy),
    False = pressed. Services: /sim/estop (SetBool: true = press, false = release) and /sim/estop_link
    (SetBool: false = stop publishing, i.e. simulate a dead receiver; true = publish again).
  * /g1/indicator/light (std_msgs/String JSON {"colour","pattern"}) and /g1/speech/say (std_msgs/String):
    log-only on the sim (the real robot's light bar / speaker are not integrated yet).
"""
import json

from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


class RobotStubs:
    def __init__(self, node, estop_rate=10.0, enabled=True):
        self.node = node
        self.pressed = False
        self.link = bool(enabled)
        self.light = None
        self.said = []
        self.pub = node.create_publisher(Bool, "/g1/estop/heartbeat", 10)
        node.create_service(SetBool, "/sim/estop", self.on_press)
        node.create_service(SetBool, "/sim/estop_link", self.on_link)
        node.create_subscription(String, "/g1/indicator/light", self.on_light, 10)
        node.create_subscription(String, "/g1/speech/say", self.on_say, 10)
        node.create_timer(1.0 / estop_rate, self.beat)

    def beat(self):
        if self.link:
            self.pub.publish(Bool(data=not self.pressed))

    def on_press(self, req, res):
        self.pressed = bool(req.data)
        res.success, res.message = True, "e-stop PRESSED" if self.pressed else "e-stop released"
        self.node.get_logger().warn(f"[sim e-stop] {res.message}")
        self.beat()
        return res

    def on_link(self, req, res):
        self.link = bool(req.data)
        res.success, res.message = True, "e-stop heartbeat on" if self.link else "e-stop heartbeat OFF (receiver dead)"
        self.node.get_logger().warn(f"[sim e-stop] {res.message}")
        return res

    def on_light(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            d = {"raw": msg.data}
        if d != self.light:
            self.light = d
            self.node.get_logger().info(f"[indicator stub] light {d.get('colour')} {d.get('pattern')}")

    def on_say(self, msg):
        self.said = (self.said + [msg.data])[-20:]
        self.node.get_logger().info(f"[speech stub] \"{msg.data}\"")
