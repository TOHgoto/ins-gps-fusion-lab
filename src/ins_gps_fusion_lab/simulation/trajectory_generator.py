"""Trajectory generator for INS-GPS fusion simulation.

TODO: Extend with circular, curved, etc. For now provides simple straight-line.
"""

from typing import Optional

import numpy as np


class TrajectoryGenerator:
    """Generate ground truth trajectories (position, velocity, acceleration).

    TODO: Add circular, figure-8, and other trajectory types.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def straight_line(
        self,
        duration: float,
        velocity: Optional[np.ndarray] = None,
        start_pos: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate straight-line trajectory at constant velocity.

        Parameters
        ----------
        duration : float
            Trajectory duration in seconds.
        velocity : np.ndarray or None
            Constant velocity (3,). Default [1, 0, 0].
        start_pos : np.ndarray or None
            Starting position (3,). Default zeros.

        Returns
        -------
        positions : np.ndarray (N, 3)
        velocities : np.ndarray (N, 3)
        accelerations : np.ndarray (N, 3)
        """
        n = int(duration / self.dt)
        t = np.arange(n) * self.dt

        vel = np.asarray(velocity if velocity is not None else [1.0, 0.0, 0.0])
        pos0 = np.asarray(start_pos if start_pos is not None else [0.0, 0.0, 0.0])

        positions = pos0 + np.outer(t, vel)
        velocities = np.tile(vel, (n, 1))
        accelerations = np.zeros((n, 3))

        return positions, velocities, accelerations
