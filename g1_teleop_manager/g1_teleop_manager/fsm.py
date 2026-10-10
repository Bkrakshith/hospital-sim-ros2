"""Mode FSM and key/velocity logic for the G1 teleop manager (docs/interfaces.md 3.2, 3.3). No ROS imports.

Legal transitions: DAMP->STAND, STAND->{LOCO, MANIP, NAV, AUTO, DAMP}, LOCO->STAND, MANIP->STAND, NAV->STAND,
AUTO->STAND, any->DAMP. LOCO->MANIP (and every other pair) is rejected. STAND->AUTO is only accepted from the
sources in AUTO_SOURCES (the mission executive or the UI, docs/interfaces.md 6). MANIP->STAND and AUTO->STAND are
accepted immediately but the new STAND reports detail "blending" for manip_blend_time while the arms go back to
default; during the blend only DAMP is accepted. Same-mode requests are accepted as no-ops.

HOLD (= "Stop", docs/ui/README.md decision 1) is kept out of LEGAL on purpose (LEGAL is the original 6-mode table):
  any mode except DAMP -> HOLD, also during an arm blend (the blend is cancelled: the arms freeze where they are);
  HOLD -> STAND (explicit resume; the arms blend from the frozen pose to default over hold_blend_time >= 2 s,
  skipped when they are already within HOLD_BLEND_TOL of default); HOLD -> AUTO only when the robot was in AUTO when
  it was stopped and the request comes from AUTO_SOURCES (resume the paused mission); HOLD -> DAMP always.
  HOLD requested in DAMP is accepted as a no-op (nothing to hold: the robot is powered down).
Sources may carry an identity after a colon ("ui:ipad-or1"); AUTO_SOURCES checks the part before it.

arms_hold (sub-state of MANIP, docs/interfaces.md 3.2): the camera teleop arm and hand streams are not forwarded and the
controller freezes the arms at the measured pose (like HOLD) while the robot stays in MANIP; switched off by the same
flag. Any mode change clears it. The gripper override (hand open/close by an operator) is only allowed in HOLD or in
MANIP with arms_hold (otherwise the camera stream owns the hands).

FORWARD says which input channel the arbiter passes through in which mode (everything else is dropped). HOLD
forwards nothing: zero velocity, no arm or hand targets (the controller freezes the arms, the hands keep their last
command).
"""
DAMP, STAND, LOCO, MANIP, NAV, AUTO, HOLD = 0, 1, 2, 3, 4, 5, 6
NAMES = {DAMP: "DAMP", STAND: "STAND", LOCO: "LOCO", MANIP: "MANIP", NAV: "NAV", AUTO: "AUTO", HOLD: "HOLD"}
LEGAL = {
    DAMP: {STAND},
    STAND: {LOCO, MANIP, NAV, AUTO, DAMP},
    LOCO: {STAND},
    MANIP: {STAND},
    NAV: {STAND},
    AUTO: {STAND},
}
AUTO_SOURCES = {"mission", "ui"}
BLEND_FROM = {MANIP, AUTO}     # leaving these to STAND blends the arms back to default
HOLD_EXITS = {STAND, AUTO, DAMP}   # AUTO only if hold_from == AUTO
MANUAL = {LOCO, MANIP}             # person drives: lease + UI heartbeat watchdog apply
MOTION = {LOCO, MANIP, NAV, AUTO}  # need the hardware e-stop heartbeat
HOLD_BLEND_TOL = 0.05              # rad: frozen arms closer than this to default -> no blend on HOLD -> STAND


def source_kind(source):
    """'ui:ipad-1' -> 'ui'; 'keyboard' -> 'keyboard'."""
    return (source or "").split(":", 1)[0]

# input channel -> modes in which it is forwarded
#   key_vel        velocity keys (w/s/a/d/q/e/space)      -> /g1/cmd_vel
#   cmd_vel_teleop /g1/cmd_vel_teleop                     -> /g1/cmd_vel
#   cmd_vel_nav    /g1/cmd_vel_nav (Nav2)                 -> /g1/cmd_vel
#   upper_teleop   /g1/teleop/upper_body_target_raw       -> /g1/upper_body_target
#   hand_teleop    /g1/teleop/hand/*/cmd_raw              -> /g1/hand/*/cmd
#   upper_auto     /g1/auto/upper_body_target             -> /g1/upper_body_target
#   hand_auto      /g1/auto/hand/*/cmd                    -> /g1/hand/*/cmd
FORWARD = {
    "key_vel": {LOCO},
    "cmd_vel_teleop": {LOCO},
    "cmd_vel_nav": {NAV, AUTO},
    "upper_teleop": {MANIP},
    "hand_teleop": {MANIP},
    "upper_auto": {AUTO},
    "hand_auto": {AUTO},
}


def forwards(mode, channel):
    """True if input `channel` drives the robot in `mode`."""
    return mode in FORWARD[channel]


class ModeFSM:
    def __init__(self, blend_time=1.0, hold_blend_time=2.5, default_upper=None):
        self.mode = DAMP
        self.source = "init"
        self.detail = "start: motors damped"
        self.blend_time = blend_time
        self.hold_blend_time = max(2.0, hold_blend_time)   # decision 1: arms blend slowly (>= 2 s) after a stop
        self.default_upper = None if default_upper is None else list(default_upper)
        self.cur_blend_time = blend_time
        self.blend_start = None        # time MANIP->STAND blend started
        self.blend_from = None         # upper-body target at MANIP exit
        self.hold_from = None          # mode the robot was in when HOLD started
        self.hold_since = None
        self.arms_hold = False
        self.changed = True

    def set_arms_hold(self, on):
        """Pause / continue the arms inside MANIP. Returns (ok, message)."""
        if on and self.mode != MANIP:
            return False, f"arms hold is only possible in MANIP (mode {NAMES[self.mode]})"
        if bool(on) == self.arms_hold:
            return True, "arms already held" if on else "arms not held"
        self.arms_hold = bool(on)
        self.detail = "arms held" if on else "ready"
        self.changed = True
        return True, "arms held (MANIP)" if on else "arms follow the operator again"

    def hand_override_allowed(self):
        return self.mode == HOLD or (self.mode == MANIP and self.arms_hold)

    def blending(self, now):
        return self.blend_start is not None and now - self.blend_start < self.cur_blend_time

    def tick(self, now):
        """Advance time-based state. Returns True if the published detail changed."""
        if self.blend_start is not None and not self.blending(now):
            self.blend_start = None
            self.blend_from = None
            if self.mode == STAND:
                self.detail = "ready"
                self.changed = True
                return True
        return False

    def request(self, mode, source, now, upper_target=None, detail=None):
        """Returns (accepted, message). upper_target: the arms' current pose (blend start when leaving MANIP/AUTO,
        or the frozen pose when leaving HOLD)."""
        res = self._request(mode, source, now, upper_target, detail)
        if self.mode != MANIP:
            self.arms_hold = False
        return res

    def _request(self, mode, source, now, upper_target=None, detail=None):
        if mode not in NAMES:
            return False, f"unknown mode {mode}"
        if mode == DAMP:
            prev = self.mode
            self.mode, self.source = DAMP, source
            self.detail = detail or ("e-stop" if source != "safety" else "fall detected")
            self.blend_start = self.blend_from = None
            self.hold_from = self.hold_since = None
            self.changed = True
            return True, f"{NAMES[prev]} -> DAMP"
        if mode == HOLD:
            return self._hold(source, now, detail)
        if mode == self.mode:
            return True, f"already {NAMES[mode]}"
        if self.mode == HOLD:
            return self._leave_hold(mode, source, now, upper_target)
        if self.blending(now):
            return False, "arms still blending back to default; wait ~1 s"
        if mode not in LEGAL[self.mode]:
            if self.mode == LOCO and mode == MANIP:
                return False, "LOCO -> MANIP rejected: pause locomotion first (p), then m"
            return False, f"{NAMES[self.mode]} -> {NAMES[mode]} not allowed"
        if mode == AUTO and source_kind(source) not in AUTO_SOURCES:
            return False, f"AUTO can only be requested by {sorted(AUTO_SOURCES)} (got '{source}')"
        prev = self.mode
        self.mode, self.source = mode, source
        self.detail = "ready" if mode != AUTO else "mission running"
        if prev in BLEND_FROM and mode == STAND:
            self.blend_start = now
            self.cur_blend_time = self.blend_time
            self.blend_from = None if upper_target is None else list(upper_target)
            self.detail = "blending arms to default"
        self.changed = True
        return True, f"{NAMES[prev]} -> {NAMES[mode]}"

    # ------------------------------------------------------------------ HOLD ("Stop")
    def _hold(self, source, now, detail=None):
        if self.mode == DAMP:
            return True, "powered off (DAMP): nothing to hold"
        if self.mode == HOLD:
            return True, "already HOLD"
        prev = self.mode
        # a stop during the arm blend back to default freezes the arms mid-blend (no motion during a handover)
        self.hold_from = STAND if (prev == STAND and self.blend_start is not None) else prev
        self.blend_start = self.blend_from = None
        self.mode, self.source = HOLD, source
        self.hold_since = now
        self.detail = detail or f"stopped (was {NAMES[prev]})"
        self.changed = True
        return True, f"{NAMES[prev]} -> HOLD"

    def _leave_hold(self, mode, source, now, upper_target):
        if mode not in HOLD_EXITS:
            return False, (f"HOLD -> {NAMES[mode]} not allowed: resume first (HOLD -> STAND), "
                           f"or resume the paused task")
        if mode == AUTO:
            if self.hold_from != AUTO:
                return False, f"HOLD -> AUTO only resumes a stopped task (robot was in {NAMES.get(self.hold_from)})"
            if source_kind(source) not in AUTO_SOURCES:
                return False, f"AUTO can only be requested by {sorted(AUTO_SOURCES)} (got '{source}')"
            self.mode, self.source, self.detail = AUTO, source, "mission resumed"
            self.hold_from = self.hold_since = None
            self.changed = True
            return True, "HOLD -> AUTO"
        # HOLD -> STAND: slow blend from the frozen pose to default (skipped if already there)
        self.mode, self.source = STAND, source
        self.hold_from = self.hold_since = None
        far = True
        if upper_target is not None and self.default_upper is not None:
            far = max(abs(a - b) for a, b in zip(upper_target, self.default_upper)) > HOLD_BLEND_TOL
        if far:
            self.blend_start = now
            self.cur_blend_time = self.hold_blend_time
            self.blend_from = None if upper_target is None else list(upper_target)
            self.detail = "resumed: blending arms to rest"
        else:
            self.detail = "ready"
        self.changed = True
        return True, "HOLD -> STAND"


class VelocityIntegrator:
    """Key-driven velocity: each key steps one axis, clamped; zero after `timeout` s without a key."""

    def __init__(self, limits, step, timeout=0.5):
        self.lim = {k: tuple(v) for k, v in limits.items()}
        self.step = dict(step)
        self.timeout = timeout
        self.v = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self.last = None

    def key(self, action, now):
        ax = {"vx_up": ("vx", 1), "vx_down": ("vx", -1), "vy_up": ("vy", 1), "vy_down": ("vy", -1),
              "wz_up": ("wz", 1), "wz_down": ("wz", -1)}.get(action)
        if action == "zero":
            self.zero()
            self.last = now
            return True
        if ax is None:
            return False
        a, sgn = ax
        lo, hi = self.lim[a]
        self.v[a] = round(min(hi, max(lo, self.v[a] + sgn * self.step[a])), 6)
        self.last = now
        return True

    def zero(self):
        self.v = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def get(self, now):
        if self.last is None or now - self.last > self.timeout:
            self.zero()
        return (self.v["vx"], self.v["vy"], self.v["wz"])


def clamp_twist(v, limits):
    return tuple(min(limits[k][1], max(limits[k][0], x)) for k, x in zip(("vx", "vy", "wz"), v))


def blend(from_, to, alpha):
    a = min(1.0, max(0.0, alpha))
    a = a * a * (3 - 2 * a)
    return [f + (t - f) * a for f, t in zip(from_, to)]


KEY_TO_MODE = {"ready": (STAND, {DAMP, HOLD}), "loco": (LOCO, None), "pause": (STAND, {LOCO, NAV, AUTO}),
               "manip": (MANIP, None), "manip_stop": (STAND, {MANIP}), "nav": (NAV, None), "estop": (DAMP, None),
               "hold": (HOLD, None)}
