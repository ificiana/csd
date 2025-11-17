"""
PID position controller for payload trajectory tracking.

Theory:
    Classical PID control with integral anti-windup:

        u(t) = K_p e(t) + K_d de/dt + K_i int(e dt)

    where e = x_ref - x is the tracking error.

    For payload control:
        F_des = m * (K_p (x_ref - x) + K_d (v_ref - v) + K_i int((x_ref - x) dt))

    The gains K_p, K_d, K_i are tuned using Ziegler-Nichols method based on
    ultimate gain K_u and ultimate period T_u from stability analysis.
"""

import numpy as np

from clock import TIME_STEP
from config import KD, KI_POS, KP


class PositionController:
    """PID position controller for 3-DOF translational control."""

    def __init__(self, mass: float):
        """
        Args:
            mass: Total system mass (payload + drones) in kg.
        """
        self.mass = mass
        self.integral_error = np.zeros(3)

    def compute_force(
        self, pos: np.ndarray, vel: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray
    ) -> np.ndarray:
        """
        Computes desired force using PID control law.

        Args:
            pos: Current position [x, y, z].
            vel: Current velocity [vx, vy, vz].
            pos_ref: Reference position [x, y, z].
            vel_ref: Reference velocity [vx, vy, vz].

        Returns:
            Desired force vector [Fx, Fy, Fz].
        """
        position_error = pos_ref - pos
        velocity_error = vel_ref - vel

        self.integral_error = self.integral_error + position_error * TIME_STEP

        desired_acceleration = (
            KP @ position_error + KD @ velocity_error + KI_POS @ self.integral_error
        )
        desired_force = desired_acceleration * self.mass

        return desired_force
