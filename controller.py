from typing import TYPE_CHECKING

import numpy as np

from clock import get_time
from drone import Drone

if TYPE_CHECKING:
    from main import Cube

HEIGHT = 5
EAST = 10  # can be negative now
NORTH = 10  # can be negative now


class ThrustController:
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.g = -9.80665
        self.mass = self.payload.mass + sum(d.mass for d in self.drones)

        # takeoff power setup (vertical)
        max_thrust = drones[0].max_thrust * 4
        self.max_takeoff = 0.75 * max_thrust
        self.max_takeoff_acc = self.max_takeoff / self.mass + self.g
        self.z_coeff = np.sqrt(3 * HEIGHT * self.max_takeoff_acc / (10 * np.sqrt(3)))

        residual_thrust = max_thrust - self.max_takeoff
        self.max_transl = residual_thrust * 0.9
        self.max_transl_acc = self.max_transl / self.mass

        # Use absolute travel distances to compute coeffs (so negative targets work)
        self.travel_e = abs(
            EAST
        )  # distance to travel in E axis from start (assumes start x=0)
        self.travel_n = abs(
            NORTH
        )  # distance to travel in N axis from start (assumes start y=0)
        self.x_coeff = (
            np.sqrt(3 * self.travel_e * self.max_transl_acc / (10 * np.sqrt(3)))
            if self.travel_e > 0
            else 0.0
        )
        self.y_coeff = (
            np.sqrt(3 * self.travel_n * self.max_transl_acc / (10 * np.sqrt(3)))
            if self.travel_n > 0
            else 0.0
        )

        # --- trajectory phase control ---
        self.phase = 0
        self.t0 = get_time()
        self.hover_times = [2.0, 2.0, 2.0]  # t1, t2, t3 hover durations

        # tolerance for positional comparisons
        self.tol = 1e-6

    def accel_poly(self, coeff: float, target: float, t: float) -> float:
        """Base acceleration polynomial (cubic smooth profile)."""
        if target <= 0 or coeff == 0:
            return 0.0
        A = 60 * coeff**3 / target**2
        B = -180 * coeff**4 / target**3
        C = 120 * coeff**5 / target**4
        return A * t + B * t**2 + C * t**3

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

        # 2: Move East (handles positive or negative EAST)
        elif self.phase == 2:
            # direction to travel: +1 if EAST > 0 else -1 (if EAST==0, we skip)
            dir_e = np.sign(EAST) if not np.isclose(EAST, 0.0) else 0.0
            # remaining distance (signed)
            rem_e = EAST - pos[0]
            arrived_e = (
                np.isclose(rem_e, 0.0, atol=self.tol)
                or (dir_e > 0 and pos[0] >= EAST - self.tol)
                or (dir_e < 0 and pos[0] <= EAST + self.tol)
            )

            if dir_e == 0.0 or arrived_e:
                # reached east target
                self.phase = 3
                self.t0 = get_time()
                a_x = a_y = 0.0
                a_z = -self.g
                print("\nReached East:", EAST)
            else:
                # accelerate in direction dir_e using travel distance (abs)
                a_x = dir_e * self.accel_poly(self.x_coeff, self.travel_e, t_global)
                a_y = 0.0
                a_z = -self.g

        # 3: Hover East
        elif self.phase == 3:
            if t_global > self.hover_times[1]:
                self.phase = 4
                self.t0 = get_time()
                print("\nHover complete, moving North")
            a_x = a_y = 0.0
            a_z = -self.g

        # 4: Move North (handles positive or negative NORTH)
        elif self.phase == 4:
            dir_n = np.sign(NORTH) if not np.isclose(NORTH, 0.0) else 0.0
            rem_n = NORTH - pos[1]
            arrived_n = (
                np.isclose(rem_n, 0.0, atol=self.tol)
                or (dir_n > 0 and pos[1] >= NORTH - self.tol)
                or (dir_n < 0 and pos[1] <= NORTH + self.tol)
            )

            if dir_n == 0.0 or arrived_n:
                self.phase = 5
                self.t0 = get_time()
                a_x = a_y = 0.0
                a_z = -self.g
                print("\nReached North:", NORTH)
            else:
                a_x = 0.0
                a_y = dir_n * self.accel_poly(self.y_coeff, self.travel_n, t_global)
                a_z = -self.g

        # 5: Hover North
        elif self.phase == 5:
            if t_global > self.hover_times[2]:
                self.phase = 6
                self.t0 = get_time()
                print("\nHover complete, landing")
            a_x = a_y = 0.0
            a_z = -self.g

        # 6: Free fall :)
        elif self.phase == 6:
            a_x = a_y = a_z = 0.0
            if np.isclose(pos[2], self.payload.h, atol=self.tol):
                print("\nLanded!")
                self.phase = 7  # end
            B = self.payload.bottom[:, 2]
            minBx = np.argmin(B)
            pen = B[minBx]
            if pen < 0:
                print("\nLanded or crashed!")
                self.phase = 7  # end
        else:
            print("\nTrajectory complete.")
            exit(0)

        # --- Apply net thrust to drones ---
        Fx = self.mass * a_x  # type: ignore
        Fy = self.mass * a_y  # type: ignore
        Fz = self.mass * a_z  # type: ignore
        for d in self.drones:
            d.set_thrust(Fx / 4, Fy / 4, Fz / 4)

        # print pos but delete last line
        print(
            f"[{get_time():.2f}s] Payload COM Position: N={pos[1]:.2f}, E={pos[0]:.2f}, D={-pos[2]:.2f}",
            end="\r",
        )
