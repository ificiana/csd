import numpy as np

from clock import TIME_STEP
from entity import Entity

# time constant for response (seconds)
# higher = slower response; lower = faster, closer to instantaneous
tau = 1

# compute incremental change (first-order lag toward target)
dt = np.clip(TIME_STEP / max(tau, 1e-6), 0.0, 1.0)


class Drone(Entity):
    mass = 1
    max_thrust = 50

    def __init__(self, index: int) -> None:
        super().__init__(f"drone_{index}")
        self.thrust = np.zeros(3)
        self.pos = np.zeros(3)
        self.t_c = np.zeros(3)

        np.random.seed(1 + index)
        self.mass = self.mass + np.random.normal(0, self.mass * 0.0001)

    def set_thrust(self, x=0.0, y=0.0, z=0.0):
        # desired (commanded) thrust vector
        self.t_c = np.array([x, y, z], dtype=float)

        # error between commanded and current thrust
        e = self.t_c - self.thrust
        self.thrust += e * dt

        # enforce maximum magnitude limit
        mag = np.linalg.norm(self.thrust)
        if mag > self.max_thrust:
            self.thrust *= self.max_thrust / mag

    @property
    def telemetry(self):
        t = super().telemetry
        t["pos"] = [*self.pos]  # type: ignore
        t["command"] = [*self.t_c]  # type: ignore
        t["thrust"] = [*self.thrust]  # type: ignore
        return t
