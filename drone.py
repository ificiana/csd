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

# compute incremental change (first-order lag toward target)
dt = np.clip(TIME_STEP / max(DRONE_TIME_CONSTANT, 1e-6), 0.0, 1.0)


class Drone(Entity):
    mass = DRONE_MASS
    max_thrust = DRONE_MAX_THRUST

    def __init__(self, index: int) -> None:
        super().__init__(f"drone_{index}")
        self.thrust = np.zeros(3)
        self.pos = np.zeros(3)
        self.t_c = np.zeros(3)

        np.random.seed(1 + index)
        self.mass = self.mass + np.random.normal(0, self.mass * DRONE_MASS_VARIANCE)

    def set_thrust(self, x=0.0, y=0.0, z=0.0):
        # desired (commanded) thrust vector
        self.t_c = np.array([x, y, z], dtype=float)

        # error between commanded and current thrust
        e = self.t_c - self.thrust
        self.thrust += e * dt

        mag = np.linalg.norm(self.thrust)

        # small magnitude noise, same direction as thrust
        if mag > 1e-6:
            noise = np.random.normal(0, DRONE_THRUST_NOISE * mag, 3)
            self.thrust += noise

        # enforce maximum magnitude limit
        if mag > self.max_thrust:
            self.thrust *= self.max_thrust / mag

    @property
    def telemetry(self):
        t = super().telemetry
        t["pos"] = [*self.pos]  # type: ignore
        t["command"] = [*self.t_c]  # type: ignore
        t["thrust"] = [*self.thrust]  # type: ignore
        return t
