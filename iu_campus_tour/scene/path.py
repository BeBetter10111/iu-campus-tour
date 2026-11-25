import numpy as np


class CatmullRomPath:
    def __init__(self, waypoints, closed=True, samples_per_segment=50):
        if len(waypoints) < 2:
            raise ValueError("Catmull-Rom path requires at least two waypoints")
        self.points = np.array(waypoints, np.float32)
        self.closed = closed
        self.n_segments = len(waypoints) if closed else len(waypoints) - 1
        self._build_length_table(samples_per_segment)

    def _control_points(self, i):
        n = len(self.points)
        if self.closed:
            return tuple(self.points[idx % n] for idx in (i - 1, i, i + 1, i + 2))
        last = n - 1
        return tuple(self.points[min(max(idx, 0), last)] for idx in (i - 1, i, i + 1, i + 2))

    def point_at_u(self, u):
        u = float(u)
        if not self.closed and u >= self.n_segments:
            return self.points[-1].copy()
        u = u % self.n_segments if self.closed else max(u, 0.0)
        segment = min(int(np.floor(u)), self.n_segments - 1)
        t = u - np.floor(u)
        P0, P1, P2, P3 = self._control_points(segment)
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2.0 * P1
            + (-P0 + P2) * t
            + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
            + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
        )

    def _build_length_table(self, k):
        k = max(1, int(k))
        self.u_samples = np.linspace(0.0, self.n_segments, self.n_segments * k + 1)
        pts = np.array([self.point_at_u(u) for u in self.u_samples], dtype=np.float32)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.cum_len = np.concatenate(([0.0], np.cumsum(seg)))
        self.total_length = float(self.cum_len[-1])

    def point_at_distance(self, s):
        s = float(s)
        s = s % self.total_length if self.closed else min(max(s, 0.0), self.total_length)
        u = np.interp(s, self.cum_len, self.u_samples)
        return self.point_at_u(u)
