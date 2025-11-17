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
        For a uniform cube with mass m_c and side length a:
            I_cube = (m_c * a^2 / 6) * I_3

        For 4 point mass drones at attachment points r_i = (±a/2, ±a/2, a/2):
            I_drones = sum_i m_i * (||r_i||^2 * I_3 - r_i ⊗ r_i)

        For symmetric configuration with total drone mass m_d:
            ||r_i||^2 = (a/2)^2 + (a/2)^2 + (a/2)^2 = 3a^2/4

            Each drone contributes to diagonal:
                I_xx = m_i * (y_i^2 + z_i^2) = m_i * (a^2/4 + a^2/4) = m_i * a^2/2
                I_yy = m_i * (x_i^2 + z_i^2) = m_i * (a^2/4 + a^2/4) = m_i * a^2/2
                I_zz = m_i * (x_i^2 + y_i^2) = m_i * (a^2/4 + a^2/4) = m_i * a^2/2

            Summing over 4 drones: I_drones = (m_d/2 * a^2) * I_3

        Combined system:
            I_total = I_cube + I_drones
                    = (m_c/6 + m_d/2) * a^2 * I_3

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

from clock import TIME_STEP, get_time, stop, time_tick
from config import (
    ATTACHMENT_ERROR_SCALE,
    ATTACHMENT_ERROR_SEED,
    ATTACHMENT_POINTS,
    CUBE_MASS,
    CUBE_SIZE,
    DRONE_MASS,
    DRONE_THRUST_NOISE,
    EAST,
    ENABLE_PROFILING,
    G_ACCELERATION,
    HEIGHT,
    MAX_TAKEOFF_FRACTION,
    MAX_TRANSL_FRACTION,
    NORTH,
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
        """
        Args:
            pos: Initial position [x, y, z] at ground level.
            topic: Telemetry channel name. Default is "cube".
        """
        super().__init__(topic)
        self.pos = pos + np.array([0, 0, self.h])
        self.moi = ((self.mass / 6 + drone_total_mass / 2) * self.size**2) * np.eye(3)
        self.orientation = R.identity()
        self.omega = np.zeros(3)
        self.alpha = np.zeros(3)
        self.acc = np.zeros(3)
        self.vel = np.zeros(3)

    @property
    def top(self):
        """
        Returns world coordinates of top attachment points.

        Returns:
            4x3 array of attachment point positions.
        """
        local_coords = np.column_stack([ATTACHMENT_POINTS * self.h, np.full(4, self.h)])
        return self.orientation.apply(local_coords) + self.pos

    @property
    def bottom(self):
        """
        Returns world coordinates of bottom corners.

        Returns:
            4x3 array of bottom corner positions.
        """
        local_coords = np.column_stack(
            [ATTACHMENT_POINTS * self.h, np.full(4, -self.h)]
        )
        return self.orientation.apply(local_coords) + self.pos

    def set_alpha(self, torques: np.ndarray):
        """
        Computes angular acceleration from external torques.

        Args:
            torques: 4x3 array of torques from each drone.
        """
        self.alpha = np.linalg.inv(self.moi) @ torques.sum(axis=0)

    def set_acc(self, forces: np.ndarray):
        """
        Computes linear acceleration from external forces.

        Args:
            forces: 4x3 array of forces from each drone (including gravity).
        """
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
            if np.linalg.norm(self.vel) > 0.1:
                print(
                    f"\n[   INFO] Ground Contact - Impact Velocity: {np.linalg.norm(self.vel):.2f} m/s"
                )
                stop()
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
drone_total_mass = sum(d.mass for d in drones)
payload = Cube(np.array([0, 0, 0]))

np.random.seed(ATTACHMENT_ERROR_SEED)
attachment_error = np.random.normal(0, ATTACHMENT_ERROR_SCALE * payload.size, (4, 3))
controller = ThrustController(payload=payload, drones=drones)


def print_initialization():
    """Prints simulation configuration banner."""
    print("=" * 80)
    print("COOPERATIVE PAYLOAD TRANSPORT SIMULATION")
    print("=" * 80)
    print()
    print("PAYLOAD CONFIGURATION:")
    print(f"  Mass:            {payload.mass:.2f} kg")
    print(f"  Size:            {payload.size:.2f} m")
    print(f"  Moment of Inertia: {payload.moi[0,0]:.4f} kg·m²")
    print()
    print("DRONE CONFIGURATION:")
    for i, d in enumerate(drones):
        print(
            f"  Drone {i}:  Mass = {d.mass:.4f} kg, Max Thrust = {d.max_thrust:.1f} N"
        )
    print(f"  Total Mass:      {drone_total_mass:.4f} kg")
    print()
    print("SYSTEM CONFIGURATION:")
    print(f"  Total Mass:      {controller.mass:.2f} kg")
    print(f"  Max Thrust:      {drones[0].max_thrust * 4:.1f} N")
    print(
        f"  Max Takeoff:     {controller.max_takeoff:.1f} N ({MAX_TAKEOFF_FRACTION*100:.0f}%)"
    )
    print(
        f"  Max Translation: {controller.max_transl:.1f} N ({MAX_TRANSL_FRACTION*100:.0f}%)"
    )
    print(f"  Max Control:     {controller.max_control:.1f} N")
    print()
    print("TRAJECTORY:")
    print(f"  Average Ascend Speed:    {controller.z_coeff:.4f} m/s")
    print(f"  Average Drift Speed:     {controller.x_coeff:.4f} m/s")
    print()
    print("WAYPOINTS:")
    print(f"  Takeoff Height:  {HEIGHT:.1f} m")
    print(f"  East Target:     {EAST:.1f} m")
    print(f"  North Target:    {NORTH:.1f} m")
    print()
    print("SIMULATION PARAMETERS:")
    print(f"  Time Step:       {TIME_STEP:.4f} s")
    print(f"  Duration:        {SIM_DURATION:.1f} s")
    print(f"  Gravity:         {G_ACCELERATION:.5f} m/s²")
    print()
    print("NOISE & ERRORS:")
    print(f"  Attachment Error Seed:  {ATTACHMENT_ERROR_SEED}")
    print(f"  Attachment Error Scale: {ATTACHMENT_ERROR_SCALE:.6f}")
    print(f"  Attachment Error RMS:   {np.linalg.norm(attachment_error) / 4:.6f} m")
    # print thrust noise and mass noise
    print(f"  Thrust Noise:           {DRONE_THRUST_NOISE:.6f} N")
    print(
        f"  Mass Noise RMS:         {np.linalg.norm(drone_total_mass - DRONE_MASS * 4) / 4:.6f} kg"
    )
    print()
    print("=" * 80)
    print("STARTING SIMULATION...")
    print("=" * 80)
    print()


print_initialization()


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

    forces = drone_thrusts + drone_masses[:, np.newaxis] * G  # type: ignore
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
