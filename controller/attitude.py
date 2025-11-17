"""
Geometric attitude controller for SO(3) orientation tracking.

Theory:
    Lee 2010 geometric control on the Special Orthogonal group SO(3).
    The control law is:

        tau = -k_R * e_R - k_omega * e_omega + omega x (I * omega)

    where the attitude error in so(3) (Lie algebra) is:
        e_R = vee((R_d^T R - R^T R_d) / 2)

    and angular velocity error is:
        e_omega = omega - R^T R_d omega_d

    The gyroscopic term omega x (I * omega) compensates for angular
    momentum coupling in the Euler equations.

    THRUST ALLOCATION:
        The desired torque tau is realized by solving a least-squares
        allocation problem mapping 4 drone forces (12-DOF) to the
        required 6-DOF wrench (force + torque):

            min ||A x - b||^2

        where x is the stacked force vector, A is the wrench mapping matrix,
        and b = [0, 0, 0, tau_x, tau_y, tau_z]^T enforces zero net force
        and produces the desired torque.

References:
    Lee, T., Leok, M., & McClamroch, N. H. (2010). Geometric tracking
    control of a quadrotor UAV on SE(3). CDC 2010.
"""

import numpy as np

from config import ATTACHMENT_POINTS, KR, KW
from utils import vee


class AttitudeController:
    """Geometric attitude controller on SO(3) with thrust allocation."""

    def __init__(self, payload):
        """
        Args:
            payload: Cube payload instance with orientation and inertia properties.
        """
        self.payload = payload
        self.integral_error = np.zeros(3)

    def compute_thrust_corrections(self) -> np.ndarray:
        """
        Computes per-drone thrust corrections using geometric attitude control.

        Returns:
            4x3 array of thrust corrections [drone_i][Fx, Fy, Fz].
        """
        R = self.payload.orientation.as_matrix()
        R_desired = np.eye(3)
        omega = self.payload.omega
        omega_desired = np.zeros(3)

        attitude_error = vee(0.5 * (R_desired.T @ R - R.T @ R_desired))

        omega_error = omega - R.T @ R_desired @ omega_desired

        torque = (
            -KR * attitude_error
            - KW * omega_error
            + np.cross(omega, self.payload.moi @ omega)
        )

        r = float(self.payload.h)
        r_vectors = ATTACHMENT_POINTS * r
        r_vectors = np.column_stack([r_vectors, np.full(4, r)])

        A = np.zeros((6, 12))
        A[0:3, ::3] = 1.0
        A[0:3, 1::3] = 1.0
        A[0:3, 2::3] = 1.0

        cols = np.arange(4) * 3
        A[3, cols] = 0
        A[3, cols + 1] = -r_vectors[:, 2]
        A[3, cols + 2] = r_vectors[:, 1]
        A[4, cols] = r_vectors[:, 2]
        A[4, cols + 1] = 0
        A[4, cols + 2] = -r_vectors[:, 0]
        A[5, cols] = -r_vectors[:, 1]
        A[5, cols + 1] = r_vectors[:, 0]
        A[5, cols + 2] = 0

        b = np.concatenate([np.zeros(3), torque])

        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            x = np.zeros(12)

        thrust_corrections = x.reshape(4, 3)

        thrust_corrections -= thrust_corrections.mean(axis=0)

        return thrust_corrections
