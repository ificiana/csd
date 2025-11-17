"""
Drone actuator model with first-order thrust dynamics.

Theory:
    Real quadrotor motors exhibit first-order lag dynamics when responding
    to thrust commands due to motor and propeller inertia. The continuous-time
    response is modeled as:

        tau * dT/dt + T = T_c

    where tau is the time constant, T is actual thrust, and T_c is commanded
    thrust. Using forward Euler discretization with time step dt:

        T(t + dt) = T(t) + (T_c - T(t)) * dt/tau

    This can be written as:
        T(t + dt) = T(t) + (T_c - T(t)) * alpha

    where alpha = min(dt/tau, 1) is the discrete response rate, clamped to
    prevent instability when dt > tau.
"""

import numpy as np

from clock import TIME_STEP
from config import (
    DRONE_MASS,
    DRONE_MASS_VARIANCE,
    DRONE_MAX_THRUST,
    DRONE_THRUST_NOISE,
    DRONE_TIME_CONSTANT,
)
from entity import Entity

discrete_response_rate = np.clip(TIME_STEP / max(DRONE_TIME_CONSTANT, 1e-6), 0.0, 1.0)


class Drone(Entity):
    """Quadrotor drone with first-order thrust dynamics and noise."""

    mass = float(DRONE_MASS)
    max_thrust = float(DRONE_MAX_THRUST)

    def __init__(self, index: int) -> None:
        """
        Args:
            index: Drone identifier (0-3) for unique random seed.
        """
        super().__init__(f"drone_{index}")
        self.thrust = np.zeros(3)
        self.pos = np.zeros(3)
        self.thrust_command = np.zeros(3)
        self.feedforward_force = np.zeros(3)
        self.feedback_force = np.zeros(3)
        self.attitude_correction = np.zeros(3)

        np.random.seed(1 + index)
        self.mass = self.mass + np.random.normal(0, self.mass * DRONE_MASS_VARIANCE)

    def set_thrust(self, x=0.0, y=0.0, z=0.0):
        """
        Commands thrust vector and updates actual thrust with first-order lag.

        Args:
            x: Thrust component in x-direction (N). Default is 0.0.
            y: Thrust component in y-direction (N). Default is 0.0.
            z: Thrust component in z-direction (N). Default is 0.0.
        """
        self.thrust_command = np.array([x, y, z], dtype=float)

        thrust_error = self.thrust_command - self.thrust
        self.thrust += thrust_error * discrete_response_rate

        thrust_magnitude = np.linalg.norm(self.thrust)

        if thrust_magnitude > 1e-6:
            noise = np.random.normal(0, DRONE_THRUST_NOISE * thrust_magnitude, 3)
            self.thrust += noise

        if thrust_magnitude > self.max_thrust:
            self.thrust *= self.max_thrust / thrust_magnitude

    @property
    def telemetry(self):
        """Extends base telemetry with drone-specific data."""
        t = super().telemetry
        t["pos"] = [*self.pos]
        t["command"] = [*self.thrust_command]
        t["thrust"] = [*self.thrust]
        t["feedforward"] = [*self.feedforward_force]
        t["feedback"] = [*self.feedback_force]
        t["attitude"] = [*self.attitude_correction]
        return t
