import numpy as np

from entity import Entity


class Drone(Entity):
    mass = 1

    def __init__(self, index: int) -> None:
        super().__init__(f"drone_{index}")
        self.thrust = np.zeros(3)
        self.pos = np.zeros(3)

    @property
    def telemetry(self):
        t = super().telemetry
        t["pos"] = [*self.pos]  # type: ignore
        t["thrust"] = [*self.thrust]  # type: ignore
        return t
