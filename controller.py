from typing import TYPE_CHECKING

import numpy as np

from drone import Drone

if TYPE_CHECKING:
    from main import Cube


HEIGHT = 10
ALPHA = -1.5
BETA = -ALPHA * HEIGHT


class ThrustController:
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.g = -9.80665
        self.mass = self.payload.mass + sum([d.mass for d in self.drones])
        self.neq = -self.mass * self.g
        self.cur = lambda: np.sum([d.thrust[2] for d in drones])
        MT = 4 * self.drones[0].max_thrust
        self.prop = (self.neq + MT) / MT

    def update(self):
        height = self.payload.pos[2]
        if height < HEIGHT / self.prop:
            T = self.neq / (self.prop - 1)
        elif height < HEIGHT * 0.8:
            T = 0
        else:
            T = self.neq

        for d in self.drones:
            d.set_thrust(z=T / 4)
