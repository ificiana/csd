import numpy as np

from entity import Entity


class Drone(Entity):
    mass = 1

    def __init__(self, index: int) -> None:
        super().__init__(f"drone_{index}")
        self.thrust = np.zeros(3)
