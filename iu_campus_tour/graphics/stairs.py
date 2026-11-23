import math

import numpy as np

from graphics.fence import oriented_box
from graphics.roads import merge_meshes

RISER = 0.15
TREAD = 0.35
LANDING = 0.9


class Stair:
    def __init__(self, center, normal, width, lift):
        """Initialize a staircase centered on a doorway."""
        self.c = np.asarray(center, dtype=np.float64)
        self.n = np.asarray(normal, dtype=np.float64)
        self.d = np.array([self.n[1], -self.n[0]])
        self.width = float(width)
        self.lift = float(lift)
        self.steps = max(1, int(round(lift / RISER)))
        self.riser = lift / self.steps
        self.depth = LANDING + (self.steps - 1) * TREAD

    def _bounds(self, j):
        if j == 0:
            return -0.02, LANDING
        return LANDING + (j - 1) * TREAD, LANDING + j * TREAD

    def height_at(self, x, z):
        """Return the walking height at a point."""
        rel = np.array([x, z]) - self.c
        a, s = float(rel @ self.d), float(rel @ self.n)
        if abs(a) > self.width / 2 or s < 0.0 or s > self.depth:
            return 0.0
        j = 0 if s < LANDING else 1 + int((s - LANDING) / TREAD)
        return self.lift - min(j, self.steps - 1) * self.riser

    def footprint(self):
        """Return the staircase footprint for obstacle avoidance."""
        w, D = self.width / 2, self.depth
        return [tuple(self.c + self.d * sa + self.n * ss) for sa, ss in ((-w, 0), (w, 0), (w, D), (-w, D))]

    def parts(self):
        """Build staircase step geometry."""
        ang = math.atan2(self.d[1], self.d[0])
        out = []
        for j in range(self.steps):
            s0, s1 = self._bounds(j)
            ctr = self.c + self.n * (s0 + s1) / 2
            out.append(oriented_box(ctr[0], ctr[1], 0.0, self.lift - j * self.riser, self.width, s1 - s0, ang))
        return out


def stairs_mesh(stairs):
    """Build all staircase meshes, or return None when empty."""
    parts = [p for s in stairs for p in s.parts()]
    return merge_meshes(parts)
