"""Validation tests for the kinematic bicycle model.

Each test compares the simulation with an analytical solution.
"""

import numpy as np

from ai_racer.vehicle import VehicleParams, kinematic_bicycle, simulate

DT = 0.01
PARAMS = VehicleParams()


def test_straight_line():
    """No steering, no acceleration: constant speed along the x-axis."""
    v0, t_end = 10.0, 2.0
    n_steps = round(t_end / DT)
    state0 = np.array([0.0, 0.0, 0.0, v0])

    traj = simulate(state0, np.zeros((n_steps, 2)), PARAMS, DT)

    np.testing.assert_allclose(traj[-1], [v0 * t_end, 0.0, 0.0, v0], atol=1e-9)


def test_constant_acceleration():
    """Straight line with constant a: x = v0*t + 0.5*a*t^2, v = v0 + a*t."""
    v0, a, t_end = 5.0, 2.0, 3.0
    n_steps = round(t_end / DT)
    state0 = np.array([0.0, 0.0, 0.0, v0])
    controls = np.tile([0.0, a], (n_steps, 1))

    traj = simulate(state0, controls, PARAMS, DT)

    x_expected = v0 * t_end + 0.5 * a * t_end**2
    v_expected = v0 + a * t_end
    np.testing.assert_allclose(traj[-1], [x_expected, 0.0, 0.0, v_expected],
                               atol=1e-9)


def test_constant_steering_circle():
    """Constant steering and speed: the CG drives on a circle with
    radius R = lr / sin(beta)."""
    delta, v0, n_steps = 0.1, 10.0, 5000
    state0 = np.array([0.0, 0.0, 0.0, v0])
    controls = np.tile([delta, 0.0], (n_steps, 1))

    traj = simulate(state0, controls, PARAMS, DT)

    beta = np.arctan(PARAMS.lr / PARAMS.wheelbase * np.tan(delta))
    radius = PARAMS.lr / np.sin(beta)
    # The center lies at distance R, 90 deg to the left of the initial
    # velocity direction (psi + beta = beta at t = 0).
    center = radius * np.array([-np.sin(beta), np.cos(beta)])
    dist = np.hypot(traj[:, 0] - center[0], traj[:, 1] - center[1])

    np.testing.assert_allclose(dist, radius, atol=1e-6)


def test_steering_is_limited():
    """Steering beyond max_steer must act like steering at max_steer."""
    state = np.array([0.0, 0.0, 0.0, 10.0])
    too_much = kinematic_bicycle(state, np.array([1.5, 0.0]), PARAMS)
    at_limit = kinematic_bicycle(state, np.array([PARAMS.max_steer, 0.0]),
                                 PARAMS)

    np.testing.assert_allclose(too_much, at_limit)
