from typing import TYPE_CHECKING
import numpy as np
from drone import Drone
from clock import get_time

if TYPE_CHECKING:
    from main import Cube

HEIGHT = 5
EAST = 10
NORTH = 10

class ThrustController:
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.g = -9.80665
        self.mass = self.payload.mass + sum(d.mass for d in self.drones)

        # takeoff power setup
        max_thrust = drones[0].max_thrust * 4
        self.max_takeoff = 0.75 * max_thrust
        self.max_takeoff_acc = self.max_takeoff / self.mass + self.g
        self.z_coeff = np.sqrt(3 * HEIGHT * self.max_takeoff_acc / (10 * np.sqrt(3)))

        residual_thrust = max_thrust - self.max_takeoff
        self.max_transl = residual_thrust * 0.9
        self.max_transl_acc = self.max_transl / self.mass
        self.x_coeff = np.sqrt(3 * EAST * self.max_transl_acc / (10 * np.sqrt(3)))
        self.y_coeff = np.sqrt(3 * NORTH * self.max_transl_acc / (10 * np.sqrt(3)))

        # --- trajectory phase control ---
        self.phase = 0
        self.t0 = get_time()
        self.hover_times = [2.0, 2.0, 2.0]  # t1, t2, t3 hover durations

    def accel_poly(self, coeff: float, target: float, t: float) -> float:
        """Base acceleration polynomial (cubic smooth profile)."""
        A = 60 * coeff ** 3 / target ** 2
        B = -180 * coeff ** 4 / target ** 3
        C = 120 * coeff ** 5 / target ** 4
        return A*t + B*t**2 + C*t**3

    def update(self):
        t_global = get_time() - self.t0
        pos = self.payload.pos
        H = HEIGHT + self.payload.h

        # --- PHASE LOGIC ---
        # 0: Takeoff
        if self.phase == 0:
            if pos[2] < H:
                a_z = self.accel_poly(self.z_coeff, HEIGHT, t_global) - self.g
            else:
                a_z = -self.g
                self.phase = 1
                self.t0 = get_time()  # reset timer
                print("\nReached height:", HEIGHT)
            a_x = a_y = 0.0

        # 1: Hover at top
        elif self.phase == 1:
            if t_global > self.hover_times[0]:
                self.phase = 2
                self.t0 = get_time()
                print("\nHover complete, moving East")
            a_x = a_y = 0.0
            a_z = -self.g

        # 2: Move East
        elif self.phase == 2:
            if pos[0] < EAST:
                a_x = self.accel_poly(self.x_coeff, EAST, t_global)
                a_y = 0.0
                a_z = -self.g
            else:
                self.phase = 3
                self.t0 = get_time()
                a_x = a_y = 0.0
                a_z = -self.g
                print("\nReached East:", EAST)

        # 3: Hover East
        elif self.phase == 3:
            if t_global > self.hover_times[1]:
                self.phase = 4
                self.t0 = get_time()
                print("\nHover complete, moving North")
            a_x = a_y = 0.0
            a_z = -self.g

        # 4: Move North
        elif self.phase == 4:
            if pos[1] < NORTH:
                a_x = 0.0
                a_y = self.accel_poly(self.y_coeff, NORTH, t_global)
                a_z = -self.g
            else:
                self.phase = 5
                self.t0 = get_time()
                a_x = a_y = 0.0
                a_z = -self.g
                print("\nReached North:", NORTH)

        # 5: Hover North
        elif self.phase == 5:
            if t_global > self.hover_times[2]:
                self.phase = 6
                self.t0 = get_time()
                print("\nHover complete, landing")
            a_x = a_y = 0.0
            a_z = -self.g

        # 6: Land
        elif self.phase == 6:
            if pos[2] > self.payload.h:
                a_z = -self.accel_poly(self.z_coeff, HEIGHT, t_global) - self.g
            else:
                a_z = 0.0
                a_x = a_y = 0.0
                print("\nLanded.")
                # done landing
            a_x = a_y = 0.0

        # --- Apply net thrust to drones ---
        Fx = self.mass * a_x # type: ignore
        Fy = self.mass * a_y # type: ignore
        Fz = self.mass * a_z # type: ignore
        for d in self.drones:
            d.set_thrust(Fx/4, Fy/4, Fz/4)
            
        # print pos but delete last line
        print(f"[{get_time():.2f}s] Payload COM Position: N={pos[1]:.2f}, E={pos[0]:.2f}, D={-pos[2]:.2f}", end='\r')
