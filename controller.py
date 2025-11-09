from typing import TYPE_CHECKING

import numpy as np

from drone import Drone
from clock import get_time

if TYPE_CHECKING:
    from main import Cube


HEIGHT = 5
# max speed (m/s)
BETA = 1


class ThrustController:
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.g = -9.80665
        self.mass = self.payload.mass + sum([d.mass for d in self.drones])
        self.neq = -self.mass * self.g
        self.cur = lambda: np.sum([d.thrust[2] for d in drones])

    def update(self):
        t = get_time()
        
        # take off stage
        if self.payload.pos[2] < HEIGHT + 0.5:
            A = 60 * BETA ** 3 / HEIGHT ** 2
            B = -180 * BETA ** 4 / HEIGHT ** 3
            C = 120 * BETA ** 5 / HEIGHT ** 4
            X = A*t**1 + B*t**2 + C*t**3
            a = X - self.g
        else:
            a = -self.g
        t = self.mass * a
        for d in self.drones:
            d.set_thrust(0.0, 0.0, t / 4)
