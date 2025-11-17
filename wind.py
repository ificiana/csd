"""
Wind model for simulation environment.

This module implements a wind disturbance model with two components:
1. Constant baseline wind - light unidirectional constant wind
2. Multidirectional gusts - Gaussian random walk for turbulent gusts

Theory:
    The wind velocity is modeled as:
        v_wind(t) = v_baseline + v_gust(t)

    where:
        v_baseline is a constant 3D vector
        v_gust follows a Gaussian random walk:
            v_gust(t + dt) = v_gust(t) + N(0, sigma^2 * dt)

    This creates realistic turbulent gusts that drift over time while
    maintaining a base wind direction.
"""

import numpy as np

from config import AIR_DENSITY


class Wind:
    """Wind disturbance model with baseline and turbulent gusts."""

    def __init__(
        self,
        baseline: np.ndarray = np.array([0.5, 0.0, 0.0]),
        gust_sigma: float = 0.3,
        seed: int = 42,
    ):
        """
        Initialize wind model.

        Args:
            baseline: Constant wind velocity vector [vx, vy, vz] in m/s.
                     Default is 0.5 m/s eastward (light breeze).
            gust_sigma: Standard deviation for Gaussian random walk in m/s/sqrt(s).
                       Controls turbulence intensity.
            seed: Random seed for reproducibility.
        """
        self.baseline = np.array(baseline, dtype=float)
        self.gust_sigma = gust_sigma
        self.gust_velocity = np.zeros(3)
        self.rng = np.random.default_rng(seed)

    def update(self, dt: float):
        """
        Update wind state using Gaussian random walk.

        Args:
            dt: Time step in seconds.
        """
        # Gaussian random walk: v(t+dt) = v(t) + N(0, sigma^2 * dt)
        noise = self.rng.normal(0, self.gust_sigma * np.sqrt(dt), 3)
        self.gust_velocity += noise

    def get_velocity(self) -> np.ndarray:
        """
        Get current total wind velocity.

        Returns:
            3D wind velocity vector [vx, vy, vz] in m/s.
        """
        return self.baseline + self.gust_velocity

    def get_force(self, area: float, drag_coeff: float = 1.0) -> np.ndarray:
        """
        Calculate wind force on object.

        Args:
            area: Cross-sectional area in m^2.
            drag_coeff: Drag coefficient (dimensionless).

        Returns:
            3D force vector [Fx, Fy, Fz] in N.
        """
        # Simplified drag force: F = 0.5 * rho * Cd * A * v^2
        v_wind = self.get_velocity()
        return 0.5 * AIR_DENSITY * drag_coeff * area * v_wind * np.abs(v_wind)
