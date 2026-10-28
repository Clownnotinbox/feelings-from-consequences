"""Stage F fast safety layer = the project's feeling core."""

from collections import deque
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"


class _FalseAlarm:
    """Wraps the marker model: during an alarm window 'keep going' reads 0.95 (an alarm with an empty sky)."""

    def __init__(self, inner, owner):
        self.inner, self.owner = inner, owner
        self.thr = getattr(inner, "thr", None)

    def step(self, x):
        pc, o = self.inner.step(x)
        if self.owner.alarm_now:
            pc = np.array(pc, float)
            pc[0] = max(pc[0], 0.95)
        return pc, o


class Safety:
    def __init__(
        self,
        env,
        markers=RUNS / "markers4_s1.fwd.pkl",
        core=RUNS / "core4_s1.runtime.npz",
        fa="fa5",
        enabled=True,
        feelings=True,
        false_alarm=None,
        delay=0,
        seed=0,
    ):
        import perception, features, markers_geom, core_runtime, closed_loop

        self.F = features
        self.trk = perception.make_tracker()
        self.st = features.FeatureState(env.agent_dt)
        self.enabled = enabled
        self.mk = markers_geom.FwdMarkerCore(str(markers))
        self.layer = closed_loop.MarkerLayer(self.mk, self.mk.thr["thr_" + fa])
        self.core = core_runtime.FeelingCore(str(core)) if feelings else None
        self.feel = {}
        self.fires = 0
        self.p_keep = 0.0
        self.false_alarm = false_alarm
        self.alarm_now = False
        self.alarms = 0
        self._fa_rng = np.random.default_rng(seed + 31)
        self._fa_until = -1.0
        self._fa_next = None
        if false_alarm:
            self.layer.core = _FalseAlarm(self.mk, self)
        self.xq = deque(maxlen=int(delay) + 1) if delay else None

    def __call__(self, o, cmd):
        sm = self.trk.step(o)
        self.last_sm = sm
        x = self.F.build(o, sm, self.st, cmd)
        if self.core is not None:
            self.feel = {k: round(float(v), 3) for k, v in self.core.step(x).items()}
        if not self.enabled:
            return cmd, False
        if self.false_alarm:
            gap, dur = self.false_alarm
            t = o["t"]
            if self._fa_next is None:
                self._fa_next = t + self._fa_rng.exponential(gap)
            if t >= self._fa_next:
                self._fa_until = t + dur
                self._fa_next = t + dur + self._fa_rng.exponential(gap)
                self.alarms += 1
            self.alarm_now = t < self._fa_until
        if self.xq is not None:
            self.xq.append(x)
            x = self.xq[0]  # the danger markers see the world `delay` steps late
        new, fired, p = self.layer.act(o["t"], sm, x, cmd, o["attitude"]["yaw"])
        self.p_keep = p
        self.fires += fired
        return new, fired
