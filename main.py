"""
Rigid body dynamics simulation for cooperative payload transport.

Theory:
    This module implements 6-DOF rigid body dynamics for a cubic payload
    being transported by multiple quadrotor drones. The dynamics follow
    Newton-Euler equations:
    
    TRANSLATIONAL MOTION:
        m * dv/dt = sum(F_ext)
        dx/dt = v
    
    where F_ext includes drone thrusts and gravity.
    
    ROTATIONAL MOTION:
        I * domega/dt = sum(tau_ext) - omega x (I * omega)
    
    where tau_ext are external torques from drone forces at attachment
    points, and the second term is the gyroscopic (Coriolis) effect.
    
    ORIENTATION INTEGRATION:
        The orientation R in SO(3) is updated using exponential map:
        R(t + dt) = R(t) * exp(omega * dt)
    
    where exp is the matrix exponential (implemented via Rodrigues formula
    in scipy.spatial.transform.Rotation.from_rotvec).
    
    MOMENT OF INERTIA:
        For a uniform cube with mass m and side length a:
        I = (m * a^2 / 6) * eye(3)
    
    GROUND CONTACT:
        Simple penetration-based contact: if any bottom corner has z < 0,
        the payload is projected back to ground level and all velocities
        are zeroed (perfectly inelastic collision).

References:
    Forward Euler integration is used for simplicity. For production code,
    consider RK4 or symplectic integrators to preserve energy.
"""

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

G = np.array([0, 0, G_ACCELERATION])


class Cube(Entity):
    """Rigid body payload with 6-DOF dynamics."""
    
    mass = CUBE_MASS
    size = CUBE_SIZE
    h = size / 2

    def __init__(self, pos: np.ndarray, topic: str = "cube") -> None:
        super().__init__(topic)
        self.pos = pos + np.array([0, 0, self.h])
        self.moi = ((1 / 6) * self.mass * self.size**2) * np.eye(3)
        self.orientation = R.identity()
        self.omega = np.zeros(3)
        self.alpha = np.zeros(3)
        self.acc = np.zeros(3)
        self.vel = np.zeros(3)

    @property
    def top(self):
        """Returns world coordinates of top attachment points."""
        local_coords = np.column_stack([ATTACHMENT_POINTS * self.h, np.full(4, self.h)])
        return self.orientation.apply(local_coords) + self.pos

    @property
    def bottom(self):
        """Returns world coordinates of bottom corners."""
        local_coords = np.column_stack([ATTACHMENT_POINTS * self.h, np.full(4, -self.h)])
        return self.orientation.apply(local_coords) + self.pos

    def set_alpha(self, torques: np.ndarray):
        """Computes angular acceleration from external torques."""
        self.alpha = np.linalg.inv(self.moi) @ torques.sum(axis=0)

    def set_acc(self, forces: np.ndarray):
        """Computes linear acceleration from external forces."""
        self.acc = (forces.sum(axis=0) + self.mass * G) / (self.mass + drone_total_mass)

    def update(self):
        """Integrates equations of motion using forward Euler."""
        self.omega += self.alpha * TIME_STEP
        dR = R.from_rotvec(self.omega * TIME_STEP)
        self.orientation *= dR

        self.vel += self.acc * TIME_STEP
        self.pos += self.vel * TIME_STEP

    def tick(self):
        """Updates physics and enforces ground contact constraint."""
        self.update()

        B = self.bottom[:, 2]
        bottom_min_idx = np.argmin(B)
        penetration = B[bottom_min_idx]
        if penetration < 0:
            self.pos -= [0, 0, penetration]
            self.alpha = np.zeros(3)
            self.acc = np.zeros(3)
            self.vel = np.zeros(3)
            self.omega = np.zeros(3)
            self.update()

    @property
    def telemetry(self):
        """Extends base telemetry with full 6-DOF state."""
        t = super().telemetry
        t["pos"] = [*self.pos]
        t["rot"] = self.orientation.as_matrix().tolist()
        t["acc"] = [*self.acc]
        t["vel"] = [*self.vel]
        t["ang_acc"] = [*self.alpha]
        t["ang_vel"] = [*self.omega]
        return t


drones = [Drone(e) for e in range(4)]
print(*[f"{d.topic}:{d.mass}" for d in drones])
drone_total_mass = sum(d.mass for d in drones)
payload = Cube(np.array([0, 0, 0]))
print(payload.mass)

np.random.seed(ATTACHMENT_ERROR_SEED)
attachment_error = np.random.normal(0, ATTACHMENT_ERROR_SCALE * payload.size, (4, 3))
controller = ThrustController(payload=payload, drones=drones)


def tick():
    """Advances simulation by one time step."""
    time_tick()
    controller.update()

    attachment_points = payload.top + attachment_error
    for d, pos in zip(drones, attachment_points):
        d.pos = pos.copy()
    
    r_vectors = attachment_points - payload.pos
    drone_masses = np.array([d.mass for d in drones])
    drone_thrusts = np.array([d.thrust for d in drones])
    
    forces = drone_thrusts + drone_masses[:, np.newaxis] * G
    torques = np.cross(r_vectors, forces)
    
    payload.set_alpha(torques)
    payload.set_acc(forces)
    payload.tick()

    for d in drones:
        d.transmit()
    payload.transmit()


COMPENSATE = False

def main():
    """Main simulation loop."""
    while True:
        if COMPENSATE:
            s = time.time_ns()
            tick()
            t = time.time_ns()
            elapsed = (t - s) / 1e9 / TIME_SCALE_FACTOR
            time.sleep(max(0.0, TIME_STEP - elapsed))
        else:
            tick()
        if get_time() >= SIM_DURATION:
            return


if __name__ == "__main__":
    if ENABLE_PROFILING:
        import cProfile
        cProfile.run("main()", "prof")
    else:
        main()
