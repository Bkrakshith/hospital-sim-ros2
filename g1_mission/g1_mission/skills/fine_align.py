"""FineAlign: close the last centimetres to a station pose (the instrument table) after Nav2 has stopped within its
goal tolerance. Proportional body-frame velocity from a StationEstimator (step 1: GroundTruthStation = CHEAT;
later a table-marker / table-edge estimate from the head camera), sent on the Nav2 channel (/g1/cmd_vel_nav),
which the manager forwards in AUTO."""
import math

from g1_mission.skills.base import Skill, Status


class FineAlign(Skill):
    name = "fine_align"

    def start(self, ctx, station=None, **goal):
        super().start(ctx)
        self.station = station or ctx.locations.get(self.p("location", "theatre_instrument_table"))
        self.ok_since = None
        self.phase = "align"
        self.err = None
        self.glide_sent = False
        self.best = float("inf")

    def tick(self, ctx):
        now = ctx.robot.now()
        if self.station is None:
            return self.fail("no station pose")
        e = ctx.station.error(self.station) if ctx.station else None
        if e is None:
            if now - self.t0 > 5.0:
                return self.fail("no station estimate")
            return Status.RUNNING
        dx, dy, dyaw = e
        self.err = e
        tol_xy, tol_yaw = self.p("tol_xy", 0.03), self.p("tol_yaw", 0.05)
        inside = abs(dx) < tol_xy and abs(dy) < tol_xy and abs(dyaw) < tol_yaw
        # fixed base (arm-skill development): the base cannot walk; the sim's glide (CHEAT) moves it exactly
        if not inside and ctx.nav is not None and getattr(ctx.nav, "exact", lambda: False)():
            if not self.glide_sent:
                st = self.station
                ctx.nav.go(st["x"], st["y"], st["yaw"])
                self.glide_sent, self.phase = True, "glide (fixed base)"
            self.message = f"fixed-base glide: {self.message if self.message else ''}"
            if ctx.nav.state in ("failed", "rejected"):
                return self.fail(f"fixed-base glide failed: {ctx.nav.msg}")
            ctx.nav.distance_remaining()
            return Status.RUNNING if now - self.t0 < self.p("timeout", 45.0) else self.fail("glide timeout")
        self.message = f"err {dx * 100:+.1f} {dy * 100:+.1f} cm {math.degrees(dyaw):+.1f} deg"
        if inside:
            ctx.robot.command(cmd_vel=(0.0, 0.0, 0.0))
            self.ok_since = self.ok_since or now
            if now - self.ok_since > self.p("settle", 1.0):
                return Status.SUCCESS
            return Status.RUNNING
        self.ok_since = None
        # divergence guard: a locomotion policy that cannot track small velocity commands (the placeholder moves
        # backwards for |vx| < ~0.3 m/s commands) must not be walked away from the table
        dist = math.hypot(dx, dy)
        self.best = min(self.best, dist)
        if dist > self.best + self.p("diverge", 0.25):
            ctx.robot.command(cmd_vel=(0.0, 0.0, 0.0))
            return self.fail(f"fine align diverging ({self.message}): the walking policy does not follow the "
                             "small velocity commands")
        if now - self.t0 > self.p("timeout", 45.0):
            ctx.robot.command(cmd_vel=(0.0, 0.0, 0.0))
            return self.fail(f"fine align timeout ({self.message})")
        k = self.p("gain", [0.8, 0.8, 1.0])
        lim = self.p("max_vel", [0.12, 0.10, 0.3])
        mn = self.p("min_vel", [0.05, 0.05, 0.12])   # below this the walking policy does not move at all

        def ax(v, g, lo, hi, tol):
            if abs(v) < tol:
                return 0.0
            u = g * v
            u = math.copysign(max(abs(u), lo), u)
            return max(-hi, min(hi, u))
        # yaw first when it is large, then translate
        wz = ax(dyaw, k[2], mn[2], lim[2], tol_yaw * 0.7)
        if abs(dyaw) > 0.15:
            vx = vy = 0.0
        else:
            vx = ax(dx, k[0], mn[0], lim[0], tol_xy * 0.7)
            vy = ax(dy, k[1], mn[1], lim[1], tol_xy * 0.7)
            # asymmetric forward floor for policies with a command offset (placeholder: policy vx = 0.4 vx - 0.13,
            # so forward motion needs vx > ~0.33 m/s commanded); 0 = off
            vf = float(self.p("vx_forward_min", 0.0))
            if vx > 0.0 and vf > 0.0:
                vx = min(max(vx, vf), max(vf, lim[0]))
        ctx.robot.command(cmd_vel=(vx, vy, wz))
        if ctx.recorder is not None:
            ctx.recorder.action(cmd_vel=(vx, vy, wz))
        return Status.RUNNING

    def stop(self, ctx):
        ctx.robot.command(cmd_vel=(0.0, 0.0, 0.0))
