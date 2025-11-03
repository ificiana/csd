# from pprint import pprint
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from clock import TIME_STEP, T, get_time, time_tick
from controller import ThrustController
from drone import Drone
from entity import Entity
from utils import cross, vec3

G = vec3(0, 0, -9.80665)


class Cube(Entity):
    mass = 10
    size = 1
    h = size / 2

    def __init__(self, pos: vec3, topic: str = "cube") -> None:
        super().__init__(topic)
        self.pos = pos + (0, 0, self.h)
        self.moi = ((1 / 6) * self.mass * self.size**2) * np.eye(3)
        self.orientation = R.identity()
        self.omega = np.zeros(3)
        self.alpha = np.zeros(3)
        self.acc = np.zeros(3)
        self.vel = np.zeros(3)

    @property
    def corners(self):
        return (
            self.orientation.apply(
                np.array(
                    [
                        vec3(a * self.h, b * self.h, self.h)
                        for (a, b) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    ]
                )
            )
            + self.pos
        )

    @property
    def bottom(self):
        return (
            self.orientation.apply(
                np.array(
                    [
                        vec3(a * self.h, b * self.h, -self.h)
                        for (a, b) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                    ]
                )
            )
            + self.pos
        )

    def set_alpha(self, torques: list):
        self.alpha = np.linalg.inv(self.moi) @ sum(torques)

    def set_acc(self, forces: list):
        self.acc = (sum(forces) + self.mass * G) / (self.mass + ds_mass)

    def update(self):
        # ROTATION
        dR = R.from_rotvec(self.omega * TIME_STEP)
        self.omega += self.alpha * TIME_STEP
        self.orientation *= dR

        # TRANSLATION
        self.vel += self.acc * TIME_STEP
        self.pos += self.vel * TIME_STEP

    def tick(self):
        self.update()

        # GROUND LAW
        B = self.bottom[:, 2]
        minBx = np.argmin(B)
        pen = B[minBx]
        if pen < 0:
            self.pos -= [0, 0, pen]
            self.alpha = np.zeros(3)
            self.acc = np.zeros(3)
            self.vel = np.zeros(3)
            self.omega = np.zeros(3)
            self.update()

    @property
    def telemetry(self):
        t = super().telemetry
        t["pos"] = [*self.pos]  # type: ignore
        t["rot"] = self.orientation.as_matrix().tolist()  # type: ignore
        t["acc"] = [*self.acc]  # type: ignore
        t["vel"] = [*self.vel]  # type: ignore
        t["ang_acc"] = [*self.alpha]  # type: ignore
        t["ang_vel"] = [*self.omega]  # type: ignore
        return t


drones = [Drone(e) for e in range(4)]
print(*[f"{d.topic}:{d.mass}" for d in drones])
ds_mass = sum(d.mass for d in drones)
payload = Cube(vec3(10, 10, 0))
print(payload.mass)

np.random.seed(0)
# 0.01% of size
err_att = np.random.normal(0, 0.0001 * payload.size, (4, 3))
controller = ThrustController(payload=payload, drones=drones)


def tick():
    time_tick()
    controller.update()

    # print(payload.corners)
    # print(payload.bottom)
    # print("comps", payload.pos)

    torques = []
    forces = []
    for r, d in zip(payload.corners + err_att, drones):
        d.pos = r.copy()
        r -= payload.pos
        f = vec3(d.thrust + d.mass * G)
        forces.append(f)
        torques.append(cross(vec3(*r), f))
    payload.set_alpha(torques)
    payload.set_acc(forces)
    payload.tick()

    for d in drones:
        d.transmit()
        # pprint(d.telemetry)
    payload.transmit()
    # pprint(payload.telemetry)

    # input()


def main():
    while True:
        s = time.time_ns()
        tick()
        t = time.time_ns()
        e = (t - s) / 1e9 / 0.7
        time.sleep(max(0.0, TIME_STEP - e))

        # run for T seconds
        if get_time() >= T:
            return


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", "prof")
