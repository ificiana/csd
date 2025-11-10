from typing import TYPE_CHECKING

import numpy as np

from drone import Drone
from clock import get_time

if TYPE_CHECKING:
    from main import Cube


HEIGHT = 5


class ThrustController:
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.g = -9.80665
        self.mass = self.payload.mass + sum([d.mass for d in self.drones])
        
        # max takeoff = 75% of max thrust
        self.max_takeoff = 0.75 * drones[0].max_thrust * 4
        self.max_takeoff_acc = self.max_takeoff / self.mass + self.g
        # z coeff for takeoff profile
        self.z_coeff = np.sqrt(3 * HEIGHT * self.max_takeoff_acc / (10 * np.sqrt(3)))
        
        print(self.max_takeoff, self.max_takeoff_acc, self.z_coeff)

    def update(self):
        t = get_time()
        
        # take off stage
        if self.payload.pos[2] < HEIGHT + self.payload.h:
            A = 60 * self.z_coeff ** 3 / HEIGHT ** 2
            B = -180 * self.z_coeff ** 4 / HEIGHT ** 3
            C = 120 * self.z_coeff ** 5 / HEIGHT ** 4
            X = A*t**1 + B*t**2 + C*t**3
            a = X - self.g
        else:
            a = -self.g
        t = self.mass * a
        for d in self.drones:
            d.set_thrust(0.0, 0.0, t / 4)
