"""Vehicle models for the AI racer.

Kinematic single-track ("bicycle") model with the reference point at the
center of gravity (CG). Assumes no tire slip, so it is only valid at low
lateral acceleration. It serves as a simple baseline before the dynamic
model with Pacejka tires.

State   s = [x, y, psi, v]
    x, y  position of the CG in the global frame [m]
    psi   yaw angle (heading), counter-clockwise from the x-axis [rad]
    v     speed of the CG [m/s]

Control u = [delta, a]
    delta front steering angle [rad]
    a     longitudinal acceleration [m/s^2]
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleParams:
    """Geometric parameters of the vehicle (placeholder values, mid-size car)."""

    lf: float = 1.2                       # distance CG -> front axle [m]
    lr: float = 1.4                       # distance CG -> rear axle [m]
    max_steer: float = np.deg2rad(30.0)   # steering limit [rad]

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


def kinematic_bicycle(state: np.ndarray, control: np.ndarray,
                      params: VehicleParams) -> np.ndarray:
    """Return the time derivative of the state, ds/dt = f(s, u)."""
    x, y, psi, v = state
    delta, a = control

    # The real steering system cannot exceed its mechanical limit.
    delta = np.clip(delta, -params.max_steer, params.max_steer)

    # Slip angle at the CG: direction of the CG velocity relative to the
    # vehicle's longitudinal axis.
    beta = np.arctan(params.lr / params.wheelbase * np.tan(delta))

    x_dot = v * np.cos(psi + beta)
    y_dot = v * np.sin(psi + beta)
    psi_dot = v / params.lr * np.sin(beta)
    v_dot = a

    return np.array([x_dot, y_dot, psi_dot, v_dot])


def rk4_step(state: np.ndarray, control: np.ndarray,
             params: VehicleParams, dt: float) -> np.ndarray:
    """Advance the state by one time step dt with 4th-order Runge-Kutta.

    The control is held constant during the step (zero-order hold).
    """
    k1 = kinematic_bicycle(state, control, params)
    k2 = kinematic_bicycle(state + 0.5 * dt * k1, control, params)
    k3 = kinematic_bicycle(state + 0.5 * dt * k2, control, params)
    k4 = kinematic_bicycle(state + dt * k3, control, params)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(state0: np.ndarray, controls: np.ndarray,
             params: VehicleParams, dt: float) -> np.ndarray:
    """Simulate a sequence of controls.

    controls has shape (N, 2). Returns the trajectory with shape (N + 1, 4),
    including the initial state.
    """
    trajectory = np.empty((len(controls) + 1, 4))
    trajectory[0] = state0
    for k, u in enumerate(controls):
        trajectory[k + 1] = rk4_step(trajectory[k], u, params, dt)
    return trajectory
