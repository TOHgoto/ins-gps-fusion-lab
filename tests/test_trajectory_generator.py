"""Unit tests for trajectory generator."""

import numpy as np

from simulation.trajectory_generator import TrajectoryGenerator


def test_straight_line_shape():
    """Output shapes are correct."""
    gen = TrajectoryGenerator(dt=0.01)
    pos, vel, acc = gen.straight_line(duration=1.0)
    n = 100  # 1.0 / 0.01
    assert pos.shape == (n, 3)
    assert vel.shape == (n, 3)
    assert acc.shape == (n, 3)


def test_straight_line_constant_velocity():
    """Velocity is constant."""
    gen = TrajectoryGenerator(dt=0.01)
    _, vel, _ = gen.straight_line(duration=1.0, velocity=[2.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(vel[0], [2.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(vel[-1], [2.0, 1.0, 0.0])


def test_straight_line_position_integration():
    """Position = start + velocity * t."""
    gen = TrajectoryGenerator(dt=0.01)
    pos, vel, _ = gen.straight_line(
        duration=1.0, velocity=[1.0, 0.0, 0.0], start_pos=[0, 0, 0]
    )
    # n_steps = 100, last t = 99*0.01 = 0.99
    np.testing.assert_array_almost_equal(pos[-1], [0.99, 0.0, 0.0])
