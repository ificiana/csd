# from pprint import pprint
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from clock import TIME_STEP, get_time, time_tick
from config import (
    ATTACHMENT_ERROR_SCALE,
    ATTACHMENT_ERROR_SEED,
    ATTACHMENT_POINTS,
    CUBE_MASS,
    CUBE_SIZE,
    ENABLE_PROFILING,
    G_ACCELERATION,
    SIM_DURATION,
    TIME_SCALE_FACTOR,
)
from controller import ThrustController
from drone import Drone
from entity import Entity
from utils import cross, vec3

G = vec3(0, 0, G_ACCELERATION)


class Cube(Entity):
    mass = CUBE_MASS
    size = CUBE_SIZE
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
    def top(self):
        return (
            self.orientation.apply(
                np.array(
                    [
                        vec3(a * self.h, b * self.h, self.h)
                        for (a, b) in ATTACHMENT_POINTS
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
                        for (a, b) in ATTACHMENT_POINTS
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
        self.omega += self.alpha * TIME_STEP
        dR = R.from_rotvec(self.omega * TIME_STEP)
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
payload = Cube(vec3(0, 0, 0))
print(payload.mass)

np.random.seed(ATTACHMENT_ERROR_SEED)
err_att = np.random.normal(0, ATTACHMENT_ERROR_SCALE * payload.size, (4, 3))
controller = ThrustController(payload=payload, drones=drones)


def tick():
    time_tick()
    controller.update()

    # print(payload.corners)
    # print(payload.bottom)
    # print("comps", payload.pos)

    torques = []
    forces = []
    for r, d in zip(payload.top + err_att, drones):
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
        e = (t - s) / 1e9 / TIME_SCALE_FACTOR
        # time.sleep(max(0.0, TIME_STEP - e))

        # run for T seconds
        if get_time() >= SIM_DURATION:
            return


if __name__ == "__main__":
    if ENABLE_PROFILING:
        import cProfile
        cProfile.run("main()", "prof")
    else:
        main()
