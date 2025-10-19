import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from clock import SIM_SPEED, TIME_STEP, T, get_time, time_tick
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
        self.normal = np.zeros(3)
        self.ground_torque = np.zeros(3)

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

    def tick(self):
        # ROTATION
        # TODO: ground law, ground damping of acc
        self.omega += self.alpha * TIME_STEP
        dR = R.from_rotvec(self.omega * TIME_STEP)
        self.orientation *= dR

        # TRANSLATION
        # TODO: ground law, ground damping of alpha
        # note, pos is of the COM
        self.vel += self.acc * TIME_STEP
        self.pos += self.vel * TIME_STEP

        # print("alpha", self.alpha)
        # print("accel", self.acc)
        # print("z coords of bottom", self.bottom[:, 2], self.pos)
        B = self.bottom[:, 2]
        # print("sink", B[B < 0])

        # Baumgarte?
        # find the index of min
        # translate pos to make that 0
        # apply max normal on minBx
        minBx = np.argmin(B)
        pen = B[minBx]
        if pen < 0:
            self.pos -= [0, 0, pen]
            r = self.bottom[minBx] - self.pos
            f = -G * (ds_mass + self.mass)
            t = cross(vec3(*r), f)
            self.normal = f
            self.ground_torque = t
        else:
            self.normal = np.zeros(3)
            self.ground_torque = np.zeros(3)
        # print("z coords of bottom", self.bottom[:, 2], self.pos)

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
ds_mass = sum(d.mass for d in drones)
payload = Cube(vec3(0, 0, 0))

np.random.seed(0)
# 0.01% of size
err_att = np.random.normal(0, 0.0001 * payload.size, (4, 3))


def tick():
    time_tick()

    # print(payload.corners)
    # print(payload.bottom)
    # print("comps", payload.pos)

    torques = [payload.ground_torque]
    forces = [payload.normal]
    for r, d in zip(payload.corners + err_att, drones):
        d.pos = r
        r -= payload.pos
        f = vec3(d.thrust + d.mass * G)
        forces.append(f)
        torques.append(cross(vec3(*r), f))
    payload.set_alpha(torques)
    payload.set_acc(forces)
    payload.tick()

    for d in drones:
        d.transmit()
    payload.transmit()

    # input()


while True:
    s = time.time_ns()
    tick()
    t = time.time_ns()
    e = (t - s) / 1e9
    time.sleep(max(0.0, TIME_STEP * SIM_SPEED - e))

    # run for T seconds
    if get_time() >= T:
        exit()
