"""Control lease, UI heartbeat watchdog, hardware e-stop monitor, motion gate and speed zones for the teleop manager
(docs/interfaces.md 3.4; docs/ui/README.md decisions 1, 2, 4). No ROS imports, so every rule is unit-tested.

Trust model: the manager is authoritative for the lease. Identity comes from the channels that carry one: the SetMode
`source` ("ui:<device_id>", "keyboard", "mission", ...), the lease service, `/g1/teleop/key_ui` (JSON with a holder)
and `/g1/ui/heartbeat`. `/g1/teleop/key` (terminal keyboard) is identity "keyboard". Anonymous streams
(/g1/cmd_vel_teleop, camera teleop targets) are the physical channels of whoever holds the lease; the UI server only
forwards them from its lease holder.
"""
import math

from g1_teleop_manager.fsm import DAMP, HOLD, MANUAL, MOTION, NAMES, NAV, AUTO, STAND, source_kind

UNGATED_SOURCES = {"mission", "safety", "watchdog", "estop_hw", "lease"}   # the robot's own executives / safety


def identity(source):
    """SetMode source -> lease identity. 'ui:ipad-1' stays as is; 'keyboard' -> 'keyboard'; 'ui' -> 'ui'."""
    return source or "service"


def needs_heartbeat(holder):
    return bool(holder) and holder.startswith("ui:")


class Lease:
    """One holder at a time. Explicit leases (acquire/steal) persist until released or `ttl` s without a renew
    (heartbeat). Implicit leases are taken when someone enters LOCO/MANIP with the lease free and dropped when the
    robot leaves LOCO/MANIP."""

    def __init__(self, ttl=30.0):
        self.ttl = ttl
        self.clear()
        self.events = []          # (kind, text) for the node to log / publish

    def clear(self):
        self.holder = ""
        self.role = ""
        self.display = ""
        self.since = None
        self.renewed = None
        self.implicit = False
        self.reason = ""

    def expire(self, now):
        if self.holder and self.renewed is not None and now - self.renewed > self.ttl:
            self.events.append(("expired", f"control lease of {self.display or self.holder} expired"))
            self.clear()
            return True
        return False

    def free(self, now):
        self.expire(now)
        return not self.holder

    def held_by(self, who, now):
        self.expire(now)
        return bool(self.holder) and self.holder == who

    def held_by_other(self, who, now):
        self.expire(now)
        return bool(self.holder) and self.holder != who

    def expires_in(self, now):
        if not self.holder or self.renewed is None:
            return 0.0
        return max(0.0, self.ttl - (now - self.renewed))

    def _take(self, who, role, display, reason, now, implicit):
        self.holder, self.role, self.display = who, role or "", display or who
        self.since = self.renewed = now
        self.implicit = implicit
        self.reason = reason or ""

    def acquire(self, who, role, display, reason, now, implicit=False):
        if not who:
            return False, "holder id required"
        self.expire(now)
        if self.holder and self.holder != who:
            return False, f"control is held by {self.display or self.holder}"
        was = self.holder == who
        self._take(who, role, display, reason, now, implicit and not (was and not self.implicit))
        if not was:
            self.events.append(("acquire", f"{self.display} took control" + (f": {reason}" if reason else "")))
        return True, "already yours" if was else "control acquired"

    def release(self, who, now):
        self.expire(now)
        if not self.holder:
            return True, "lease already free"
        if self.holder != who:
            return False, f"control is held by {self.display or self.holder}"
        self.events.append(("release", f"{self.display} released control"))
        self.clear()
        return True, "control released"

    def steal(self, who, role, display, reason, now):
        if not who:
            return False, "holder id required", None
        if not (reason or "").strip():
            return False, "a reason is required to take control", None
        self.expire(now)
        prev = (self.holder, self.display) if self.holder and self.holder != who else None
        self._take(who, role, display, reason, now, False)
        txt = f"{self.display} took control" + (f" from {prev[1] or prev[0]}" if prev else "") + f": {reason}"
        self.events.append(("steal", txt))
        return True, txt, prev

    def renew(self, who, now):
        if self.holder and self.holder == who:
            self.renewed = now
            return True
        return False

    def as_dict(self, now):
        if not self.holder:
            return {"holder": "", "role": "", "display": "", "since": None, "expires_in": 0.0, "implicit": False}
        return {"holder": self.holder, "role": self.role, "display": self.display,
                "held_for": round(now - self.since, 1), "expires_in": round(self.expires_in(now), 1),
                "implicit": self.implicit, "reason": self.reason, "heartbeat": needs_heartbeat(self.holder)}


class Watchdog:
    """In LOCO/MANIP, a heartbeat-watched lease holder (ui:*) that has not sent /g1/ui/heartbeat for > timeout s
    trips the robot into HOLD (source "watchdog")."""

    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self.last = {}            # holder -> time of last heartbeat

    def beat(self, holder, now):
        self.last[holder] = now

    def age(self, holder, now):
        t = self.last.get(holder)
        return None if t is None else now - t

    def tripped(self, mode, lease, now):
        if mode not in MANUAL or not needs_heartbeat(lease.holder):
            return False
        t = self.last.get(lease.holder)
        # grace: a lease taken without any heartbeat yet counts from the moment it was taken
        if t is None or (lease.since is not None and t < lease.since):
            t = lease.since if lease.since is not None else now
        return now - t > self.timeout


class EstopMonitor:
    """Hardware wireless e-stop heartbeat (/g1/estop/heartbeat, std_msgs/Bool at 10 Hz; True = released / healthy,
    False = pressed). Missing for > timeout s or pressed -> motion locked. `required=False` is the DEV-ONLY override."""

    def __init__(self, timeout=0.5, required=True):
        self.timeout = timeout
        self.required = required
        self.last = None
        self.pressed = False
        self.ever_seen = False
        self.press_count = 0

    def update(self, released, now):
        was = self.pressed
        self.pressed = not bool(released)
        if self.pressed and not was:
            self.press_count += 1
        self.last = now
        self.ever_seen = True
        return self.pressed and not was       # True on a new press

    def alive(self, now):
        return self.last is not None and now - self.last <= self.timeout

    def ok(self, now):
        if not self.required:
            return True
        return self.alive(now) and not self.pressed

    def reason(self, now):
        if not self.required:
            return ""
        if self.last is None:
            return "hardware e-stop not seen (no /g1/estop/heartbeat)"
        if not self.alive(now):
            return f"hardware e-stop heartbeat lost ({now - self.last:.1f} s)"
        if self.pressed:
            return "hardware e-stop pressed: release it when safe"
        return ""

    def as_dict(self, now):
        return {"ok": self.ok(now), "alive": self.alive(now), "pressed": self.pressed,
                "age": None if self.last is None else round(now - self.last, 2), "presses": self.press_count,
                "dev_override": not self.required, "reason": self.reason(now)}


def gate(target, source, cur_mode, lease, estop, now):
    """Interlocks in front of ModeFSM.request. Returns (ok, message). Stop (HOLD) and power off (DAMP) always pass.
    `source` is the SetMode source (identity for the lease)."""
    if target in (HOLD, DAMP):
        return True, ""
    who = identity(source)
    kind = source_kind(source)
    if target in MOTION and target != cur_mode and not estop.ok(now):
        return False, f"motion locked: {estop.reason(now)}"
    if cur_mode == HOLD and target in (STAND, AUTO) and estop.required and estop.pressed:
        return False, "hardware e-stop pressed: release it when safe, then resume"
    if target in MANUAL and target != cur_mode:
        if lease.held_by_other(who, now):
            return False, f"controlled by {lease.display or lease.holder}: take control first"
    elif target in (NAV, AUTO) and target != cur_mode and kind not in UNGATED_SOURCES:
        if lease.held_by_other(who, now):
            return False, f"controlled by {lease.display or lease.holder}: ask them or take control"
    return True, ""


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def zone_at(x, y, zones):
    """First zone (config order = priority) containing (x, y), or None. zones: [{name, max_speed, polygon}]"""
    for z in zones or []:
        if point_in_polygon(x, y, z["polygon"]):
            return z
    return None


def clamp_speed(v, cap, wz_cap=None):
    """Scale (vx, vy) so its norm <= cap (keeps direction); clamp |wz| <= wz_cap. cap None = no clamp."""
    vx, vy, wz = v
    if cap is not None:
        n = math.hypot(vx, vy)
        if n > cap > 0:
            vx, vy = vx * cap / n, vy * cap / n
        elif cap <= 0:
            vx = vy = 0.0
    if wz_cap is not None:
        wz = max(-wz_cap, min(wz_cap, wz))
    return (vx, vy, wz)


__all__ = ["Lease", "Watchdog", "EstopMonitor", "gate", "zone_at", "clamp_speed", "point_in_polygon", "identity",
           "needs_heartbeat", "NAMES"]
