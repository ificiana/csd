from typing import TYPE_CHECKING

import numpy as np

from clock import TIME_STEP, get_time
from drone import Drone
from tune import DOFOscillationTracker

if TYPE_CHECKING:
    from main import Cube

HEIGHT = 5
EAST = 10
NORTH = 10


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# 4x3 thrust mixing matrix
# + + +
# - + -
# - - +
# + - -
MX = np.array(
    [
        [1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, -1],
    ]
)

KR = np.diag([8.0] * 3)
KW = np.diag([1.5] * 3)
KI = np.diag([0.1] * 3)


# Ziegler-Nichols tuned position controller gains
Kuxy = 4
Tuxy = 3.14
Kuz = 6
Tuz = 2.56

KP = np.diag([0.8 * Kuxy, 0.8 * Kuxy, 0.8 * Kuz])
KI_POS = np.diag([0 * Kuxy / Tuxy, 0 * Kuxy / Tuxy, 0 * Kuz / Tuz])
KD = np.diag([0.1 * Kuxy * Tuxy, 0.1 * Kuxy * Tuxy, 0.1 * Kuz * Tuz])

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

        self.max_control = residual_thrust - self.max_transl

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
        self.hover_times = [2.0, 2.0, 10.0]  # t1, t2, t3 hover durations

        # tolerance for positional comparisons
        self.tol = 1e-6

        # Attitude control integrator state
        self.int_eR = np.zeros(3)
        # Position control integrator state
        self.int_eX = np.zeros(3)

        self.tracker = DOFOscillationTracker()

    def pos_poly(self, coeff: float, target: float, t: float) -> float:
        """Base position polynomial (quintic smooth profile)."""
        if target <= 0 or coeff == 0:
            return 0.0
        A = 10 * coeff**3 / target**2
        B = -15 * coeff**4 / target**3
        C = 6 * coeff**5 / target**4
        return A * t**3 + B * t**4 + C * t**5

    def vel_poly(self, coeff: float, target: float, t: float) -> float:
        """Base velocity polynomial (quartic smooth profile)."""
        if target <= 0 or coeff == 0:
            return 0.0
        A = 30 * coeff**3 / target**2
        B = -60 * coeff**4 / target**3
        C = 30 * coeff**5 / target**4
        return A * t**2 + B * t**3 + C * t**4

    def accel_poly(self, coeff: float, target: float, t: float) -> float:
        """Base acceleration polynomial (cubic smooth profile)."""
        if target <= 0 or coeff == 0:
            return 0.0
        A = 60 * coeff**3 / target**2
        B = -180 * coeff**4 / target**3
        C = 120 * coeff**5 / target**4
        return A * t + B * t**2 + C * t**3

    def reference_pos(self):
        # as per current trajectory phase
        t = get_time() - self.t0
        H = HEIGHT + self.payload.h
        match self.phase:
            case 0:
                return np.array([0.0, 0.0, self.pos_poly(self.z_coeff, H, t)])
            case 1:
                return np.array([0.0, 0.0, H])
            case 2:
                return np.array([self.pos_poly(self.x_coeff, self.travel_e, t), 0.0, H])
            case 3:
                return np.array([EAST, 0.0, H])
            case 4:
                return np.array(
                    [EAST, self.pos_poly(self.y_coeff, self.travel_n, t), H]
                )
            case 5:
                return np.array([EAST, NORTH, H])
            case 6:
                return np.array([EAST, NORTH, 0.0])

    def reference_vel(self):
        # as per current trajectory phase
        t = get_time() - self.t0
        H = HEIGHT + self.payload.h
        match self.phase:
            case 0:
                return np.array([0.0, 0.0, self.vel_poly(self.z_coeff, H, t)])
            case 1:
                return np.array([0.0, 0.0, 0.0])
            case 2:
                return np.array(
                    [self.vel_poly(self.x_coeff, self.travel_e, t), 0.0, 0.0]
                )
            case 3:
                return np.array([0.0, 0.0, 0.0])
            case 4:
                return np.array(
                    [0.0, self.vel_poly(self.y_coeff, self.travel_n, t), 0.0]
                )
            case 5:
                return np.array([0.0, 0.0, 0.0])
            case 6:
                return np.array([0.0, 0.0, 0.0])

    def attitude_control(self):
        R = self.payload.orientation.as_matrix()
        Rd = np.eye(3)
        eR = vee(0.5 * (Rd.T @ R - R.T @ Rd))
        ew = self.payload.omega - R.T @ Rd @ np.zeros(3)

        self.int_eR = self.int_eR + eR * TIME_STEP
        tau = -KR @ eR - KW @ ew - KI @ self.int_eR

        # shapes
        # print("Ac", eR.shape, ew.shape, self.int_eR.shape, tau.shape)
        return tau

    def position_control(self):
        # use reference position and velocity to compute desired force
        pos = self.payload.pos
        vel = self.payload.vel
        pos_ref = self.reference_pos()
        vel_ref = self.reference_vel()
        e_pos = pos_ref - pos
        e_vel = vel_ref - vel # type: ignore

        self.int_eX = self.int_eX + e_pos * TIME_STEP
        a_des = KP @ e_pos + KD @ e_vel + KI_POS @ self.int_eX
        F_des = a_des * self.mass

        # shapes
        # print("X", e_pos.shape, e_vel.shape, a_des.shape, F_des.shape)
        return F_des

    def attitude_update(self):
        tau = self.attitude_control()
        arm_len = np.sqrt(3 * self.payload.h**2)
        d_rot = np.tile(tau / arm_len, (3, 1))
        # make 4x3  matrix
        dT = -MX @ d_rot
        # print(f"Attitude Control Torque: {tau}, dT: {dT}")

        # print("Au", tau.shape, d_rot.shape, dT.shape)
        return dT

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
            # if t_global > self.hover_times[2]:
            #     self.phase = 6
            #     self.t0 = get_time()
            #     print("\nHover complete, landing")
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
        F = self.mass * np.array([a_x, a_y, a_z])

        # dT = self.attitude_update()
        dT = np.zeros((4, 3))
        dF = self.position_control()

        # dF = np.zeros(3)
        # print(F, dF, dT.sum(axis=0))
        # limit combined control inputs to max available thrust
        total_control = np.linalg.norm(dF + dT.sum(axis=0))
        if total_control > self.max_control:
            scale = self.max_control / total_control
            dF = dF * scale
            dT = dT * scale
        F = dT + (F + dF) / 4
        # print(F)

        for d, F_i in zip(self.drones, F):
            d.set_thrust(*F_i)

        # track DOF oscillations
        t = get_time()
        yaw, pitch, roll = self.payload.orientation.as_euler("ZYX", degrees=True)

        axes = {
            "N": pos[1],
            "E": pos[0],
            "D": -pos[2],
            "Roll": roll,
            "Pitch": pitch,
            "Yaw": yaw,
        }
        for name, value in axes.items():
            self.tracker.update(name, value, t)

        # print(
        #     f"[{get_time():.2f}s] Payload COM Position: N={pos[1]:.3f}, E={pos[0]:.3f}, D={-pos[2]:.3f}, Yaw={yaw:.1f}, Pitch={pitch:.1f}, Roll={roll:.1f}",
        #     end="\r",
        # )

        if self.phase > 7:
            e_info = self.tracker.get_info("D")
            
            # clear line
            # print(" " * 150, end="\r")
            print(
                f"[{t:.2f}s] Payload COM Position: "
                f"N={pos[1]:.3f}, E={pos[0]:.3f}, D={-pos[2]:.3f} | "
                f"max={e_info['max']:.3f} at {e_info['t_max']:.2f}s, "
                f"min={e_info['min']:.3f} at {e_info['t_min']:.2f}s | "
                f"period={e_info['period']:.2f}s",
                f"aplitude={e_info['current_amplitude']:.3f}",
                end="\r",
            )

        else:
            print(
                f"[{get_time():.2f}s] Payload COM Position: N={pos[1]:.3f}, "
                f"E={pos[0]:.3f}, D={-pos[2]:.3f}, Yaw={yaw:.1f}, "
                f"Pitch={pitch:.1f}, Roll={roll:.1f}",
                end="\r",
            )
