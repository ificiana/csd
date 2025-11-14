"""
Trajectory planning with multi-phase state machine.

Theory:
    The mission is decomposed into discrete phases:
    0. Takeoff - Vertical ascent to target altitude
    1. Hover - Stabilize at altitude
    2. Translate East - Horizontal motion along x-axis
    3. Hover - Stabilize at waypoint
    4. Translate North - Horizontal motion along y-axis
    5. Hover - Final position hold
    6. Land - Controlled descent (optional)

    Each phase generates reference position, velocity, and acceleration
    trajectories using quintic polynomials for smooth transitions.
"""

import numpy as np

from clock import get_time
from config import (
    EAST,
    HEIGHT,
    HOVER_TIMES,
    NORTH,
    POSITION_TOLERANCE,
    G_ACCELERATION as gravity,
)
from controller import quintic


class TrajectoryPlanner:
    """Multi-phase trajectory state machine with quintic motion profiles."""

    def __init__(
        self,
        payload_half_height: float,
        z_coeff: float,
        x_coeff: float,
        y_coeff: float,
        travel_east: float,
        travel_north: float,
    ):
        self.payload_h = payload_half_height
        self.z_coeff = z_coeff
        self.x_coeff = x_coeff
        self.y_coeff = y_coeff
        self.travel_e = travel_east
        self.travel_n = travel_north

        self.phase = 0
        self.phase_start_time = get_time()
        self.hover_times = HOVER_TIMES
        self.tolerance = POSITION_TOLERANCE

    def get_reference_position(self) -> np.ndarray:
        """Returns reference position for current trajectory phase."""
        t = get_time() - self.phase_start_time
        H = HEIGHT + self.payload_h

        match self.phase:
            case 0:
                return np.array([0.0, 0.0, quintic.position(self.z_coeff, H, t)])
            case 1:
                return np.array([0.0, 0.0, H])
            case 2:
                return np.array(
                    [quintic.position(self.x_coeff, self.travel_e, t), 0.0, H]
                )
            case 3:
                return np.array([EAST, 0.0, H])
            case 4:
                return np.array(
                    [EAST, quintic.position(self.y_coeff, self.travel_n, t), H]
                )
            case 5:
                return np.array([EAST, NORTH, H])
            case 6:
                z_land = H - quintic.position(self.z_coeff, H - self.payload_h, t)
                return np.array([EAST, NORTH, z_land])
            case _:
                return np.array([EAST, NORTH, self.payload_h])

    def get_reference_velocity(self) -> np.ndarray:
        """Returns reference velocity for current trajectory phase."""
        t = get_time() - self.phase_start_time
        H = HEIGHT + self.payload_h

        match self.phase:
            case 0:
                return np.array([0.0, 0.0, quintic.velocity(self.z_coeff, H, t)])
            case 1:
                return np.array([0.0, 0.0, 0.0])
            case 2:
                return np.array(
                    [quintic.velocity(self.x_coeff, self.travel_e, t), 0.0, 0.0]
                )
            case 3:
                return np.array([0.0, 0.0, 0.0])
            case 4:
                return np.array(
                    [0.0, quintic.velocity(self.y_coeff, self.travel_n, t), 0.0]
                )
            case 5:
                return np.array([0.0, 0.0, 0.0])
            case 6:
                vz_land = -quintic.velocity(self.z_coeff, H - self.payload_h, t)
                return np.array([0.0, 0.0, vz_land])
            case _:
                return np.array([0.0, 0.0, 0.0])

    def get_feedforward_acceleration(
        self, pos: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Computes feedforward acceleration for current phase.

        Args:
            pos: Current payload position.
            gravity: Gravity acceleration (negative).

        Returns:
            Tuple of (a_x, a_y, a_z) accelerations.
        """
        t_global = get_time() - self.phase_start_time
        H = HEIGHT + self.payload_h

        match self.phase:
            case 0:
                if pos[2] < H:
                    a_z = quintic.acceleration(self.z_coeff, HEIGHT, t_global) - gravity
                else:
                    a_z = -gravity
                    self._advance_phase(1, "Reached height: {}", HEIGHT)
                a_x = a_y = 0.0

            case 1:
                if t_global > self.hover_times[0]:
                    self._advance_phase(2, "Hover complete, moving East")
                a_x = a_y = 0.0
                a_z = -gravity

            case 2:
                dir_e = np.sign(EAST) if not np.isclose(EAST, 0.0) else 0.0
                rem_e = EAST - pos[0]
                arrived_e = (
                    np.isclose(rem_e, 0.0, atol=self.tolerance)
                    or (dir_e > 0 and pos[0] >= EAST - self.tolerance)
                    or (dir_e < 0 and pos[0] <= EAST + self.tolerance)
                )

                if dir_e == 0.0 or arrived_e:
                    self._advance_phase(3, "Reached East: {}", EAST)
                    a_x = a_y = 0.0
                    a_z = -gravity
                else:
                    a_x = dir_e * quintic.acceleration(
                        self.x_coeff, self.travel_e, t_global
                    )
                    a_y = 0.0
                    a_z = -gravity

            case 3:
                if t_global > self.hover_times[1]:
                    self._advance_phase(4, "Hover complete, moving North")
                a_x = a_y = 0.0
                a_z = -gravity

            case 4:
                dir_n = np.sign(NORTH) if not np.isclose(NORTH, 0.0) else 0.0
                rem_n = NORTH - pos[1]
                arrived_n = (
                    np.isclose(rem_n, 0.0, atol=self.tolerance)
                    or (dir_n > 0 and pos[1] >= NORTH - self.tolerance)
                    or (dir_n < 0 and pos[1] <= NORTH + self.tolerance)
                )

                if dir_n == 0.0 or arrived_n:
                    self._advance_phase(5, "Reached North: {}", NORTH)
                    a_x = a_y = 0.0
                    a_z = -gravity
                else:
                    a_x = 0.0
                    a_y = dir_n * quintic.acceleration(
                        self.y_coeff, self.travel_n, t_global
                    )
                    a_z = -gravity

            case 5:
                if t_global > self.hover_times[2]:
                    self._advance_phase(6, "Hover complete, initiating soft landing")
                a_x = a_y = 0.0
                a_z = -gravity

            case 6:
                H = HEIGHT + self.payload_h
                landing_distance = H - self.payload_h
                if pos[2] <= self.payload_h + self.tolerance:
                    print("\n" + "=" * 80)
                    print("PAYLOAD LANDED SUCCESSFULLY!")
                    print("=" * 80)
                    self.phase = 7
                    a_x = a_y = a_z = 0.0
                else:
                    a_x = a_y = 0.0
                    a_z = (
                        -quintic.acceleration(self.z_coeff, landing_distance, t_global)
                        - gravity
                    )

            case _:
                print("\n" + "=" * 80)
                print("SIMULATION COMPLETE")
                print("=" * 80)
                exit(0)

        return a_x, a_y, a_z

    def _advance_phase(self, new_phase: int, message: str, *args):
        """Advances to next phase and resets timer."""
        self.phase = new_phase
        self.phase_start_time = get_time()
        print(f"\n[PHASE {new_phase}] " + message.format(*args))

    def check_ground_contact(self, bottom_corners: np.ndarray) -> bool:
        """
        Checks if payload has contacted ground during landing phase.

        Args:
            bottom_corners: Array of bottom corner z-coordinates.

        Returns:
            True if ground contact detected.
        """
        if self.phase != 6:
            return False

        B = bottom_corners[:, 2]
        bottom_min_idx = np.argmin(B)
        penetration = B[bottom_min_idx]
        if penetration < 0:
            print("\n" + "!" * 80)
            print("GROUND CONTACT DETECTED - CRASH OR HARD LANDING!")
            print("!" * 80)
            self.phase = 7
            return True
        return False
