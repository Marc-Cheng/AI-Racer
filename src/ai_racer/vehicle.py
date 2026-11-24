"""Vehicle models for the AI racer.

Two single-track ("bicycle") models with the reference point at the center
of gravity (CG). Both use the same control input, so they are interchangeable.

Control u = [delta, a]
    delta front steering angle [rad]
    a     longitudinal acceleration command [m/s^2]

1. Kinematic model: no tire slip, only valid at low lateral acceleration.
   State s = [x, y, psi, v]
       x, y  position of the CG in the global frame [m]
       psi   yaw angle (heading), counter-clockwise from the x-axis [rad]
       v     speed of the CG [m/s]

2. Dynamic model: lateral tire forces from the Pacejka magic formula.
   State s = [x, y, psi, vx, vy, r]
       vx, vy  velocity of the CG in the vehicle (body) frame [m/s]
       r       yaw rate [rad/s]
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleParams:
    """Vehicle parameters (placeholder values, sporty mid-size car)."""

    # Geometry
    lf: float = 1.2                       # distance CG -> front axle [m]
    lr: float = 1.4                       # distance CG -> rear axle [m]
    max_steer: float = np.deg2rad(30.0)   # steering limit [rad]

    # Mass properties
    m: float = 1200.0                     # mass [kg]
    iz: float = 2000.0                    # yaw moment of inertia [kg m^2]
    g: float = 9.81                       # gravity [m/s^2]

    # Pacejka lateral tire coefficients (same tire front and rear)
    mu: float = 1.0                       # friction coefficient [-]
    b: float = 10.0                       # stiffness factor [-]
    c: float = 1.9                        # shape factor [-]
    e: float = 0.97                       # curvature factor [-]

    # Below this speed the slip angles are ill-defined (division by vx).
    v_min: float = 1.0                    # [m/s]

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


def pacejka_lateral(alpha: float, fz: float, params: VehicleParams) -> float:
    """Lateral tire force [N] from the Pacejka magic formula.

    alpha is the slip angle [rad], fz the vertical load on the tire [N].
    A positive slip angle gives a positive (leftward) force.
    """
    d = params.mu * fz                    # peak force
    ba = params.b * alpha
    return d * np.sin(params.c * np.arctan(ba - params.e * (ba - np.arctan(ba))))


def dynamic_bicycle(state: np.ndarray, control: np.ndarray,
                    params: VehicleParams) -> np.ndarray:
    """Return the time derivative of the state, ds/dt = f(s, u).

    Simplifications: static axle loads (no load transfer), drive force on
    the rear axle as Fx = m * a, no combined slip, no aerodynamic drag.
    """
    x, y, psi, vx, vy, r = state
    delta, a = control
    p = params

    delta = np.clip(delta, -p.max_steer, p.max_steer)

    # Avoid division by ~0 in the slip angles at very low speed.
    vx_safe = max(vx, p.v_min)

    # Slip angles: wheel heading minus direction of the wheel velocity.
    # The axle velocity has an extra lateral part from the rotation (+-l*r).
    alpha_f = delta - np.arctan2(vy + p.lf * r, vx_safe)
    alpha_r = -np.arctan2(vy - p.lr * r, vx_safe)

    # Static load split: the axle closer to the CG carries more weight.
    fz_f = p.m * p.g * p.lr / p.wheelbase
    fz_r = p.m * p.g * p.lf / p.wheelbase

    fy_f = pacejka_lateral(alpha_f, fz_f, p)
    fy_r = pacejka_lateral(alpha_r, fz_r, p)
    fx_r = p.m * a

    # Newton-Euler in the rotating body frame. The terms +vy*r and -vx*r
    # appear because the frame itself rotates with the car.
    vx_dot = (fx_r - fy_f * np.sin(delta)) / p.m + vy * r
    vy_dot = (fy_r + fy_f * np.cos(delta)) / p.m - vx * r
    r_dot = (p.lf * fy_f * np.cos(delta) - p.lr * fy_r) / p.iz

    # Rotate the body-frame velocity into the global frame.
    x_dot = vx * np.cos(psi) - vy * np.sin(psi)
    y_dot = vx * np.sin(psi) + vy * np.cos(psi)
    psi_dot = r

    return np.array([x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot])


def rk4_step(state: np.ndarray, control: np.ndarray, params: VehicleParams,
             dt: float, model=kinematic_bicycle) -> np.ndarray:
    """Advance the state by one time step dt with 4th-order Runge-Kutta.

    The control is held constant during the step (zero-order hold).
    """
    k1 = model(state, control, params)
    k2 = model(state + 0.5 * dt * k1, control, params)
    k3 = model(state + 0.5 * dt * k2, control, params)
    k4 = model(state + dt * k3, control, params)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(state0: np.ndarray, controls: np.ndarray, params: VehicleParams,
             dt: float, model=kinematic_bicycle) -> np.ndarray:
    """Simulate a sequence of controls.

    controls has shape (N, 2). Returns the trajectory with shape
    (N + 1, n_states), including the initial state.
    """
    trajectory = np.empty((len(controls) + 1, len(state0)))
    trajectory[0] = state0
    for k, u in enumerate(controls):
        trajectory[k + 1] = rk4_step(trajectory[k], u, params, dt, model)
    return trajectory
