"""
Main thrust controller coordinating all control loops.

Integrates trajectory planning, position control, attitude control, and
manages thrust allocation to individual drones with saturation limits.
"""

from typing import TYPE_CHECKING

import numpy as np

from clock import get_time
from config import EAST, G_ACCELERATION as G
from config import (
    HEIGHT,
    MAX_TAKEOFF_FRACTION,
    MAX_TRANSL_FRACTION,
    NORTH,
)
from controller.attitude import AttitudeController
from controller.plan import TrajectoryPlanner
from controller.position import PositionController
from drone import Drone
from tune import DOFOscillationTracker

if TYPE_CHECKING:
    from main import Cube


class ThrustController:
    """Hierarchical controller combining geometric attitude and PID position control."""
    
    def __init__(self, drones: list[Drone], payload: "Cube") -> None:
        self.drones = drones
        self.payload = payload
        self.mass = self.payload.mass + sum(d.mass for d in self.drones)
        
        max_thrust = drones[0].max_thrust * 4
        self.max_takeoff = MAX_TAKEOFF_FRACTION * max_thrust
        self.max_takeoff_acc = self.max_takeoff / self.mass + G
        self.z_coeff = np.sqrt(3 * HEIGHT * self.max_takeoff_acc / (10 * np.sqrt(3)))
        
        residual_thrust = max_thrust - self.max_takeoff
        self.max_transl = residual_thrust * MAX_TRANSL_FRACTION
        self.max_transl_acc = self.max_transl / self.mass
        
        self.max_control = residual_thrust - self.max_transl
        
        self.travel_e = abs(EAST)
        self.travel_n = abs(NORTH)
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
        
        self.planner = TrajectoryPlanner(
            self.payload.h, self.z_coeff, self.x_coeff, self.y_coeff,
            self.travel_e, self.travel_n
        )
        self.position_controller = PositionController(self.mass)
        self.attitude_controller = AttitudeController(self.payload)
        
        self.tracker = DOFOscillationTracker()
    
    def update(self):
        """Updates controller state and computes thrust commands for current phase."""
        pos = self.payload.pos
        
        a_x, a_y, a_z = self.planner.get_feedforward_acceleration(pos, G)
        
        self.planner.check_ground_contact(self.payload.bottom)
        
        feedforward_force = self.mass * np.array([a_x, a_y, a_z])
        thrust_corrections = self.attitude_controller.compute_thrust_corrections()
        feedback_force = self.position_controller.compute_force(
            self.payload.pos,
            self.payload.vel,
            self.planner.get_reference_position(),
            self.planner.get_reference_velocity()
        )
        
        total_control = np.linalg.norm(feedback_force + thrust_corrections.sum(axis=0))
        if total_control > self.max_control:
            scale = self.max_control / total_control
            feedback_force = feedback_force * scale
            thrust_corrections = thrust_corrections * scale
        
        per_drone_thrust = thrust_corrections + (feedforward_force + feedback_force) / 4
        
        for d, F_i in zip(self.drones, per_drone_thrust):
            d.set_thrust(*F_i)
        
        self._update_telemetry(pos)
    
    def _update_telemetry(self, pos: np.ndarray):
        """Updates oscillation tracking and prints live telemetry."""
        t = get_time()
        yaw, pitch, roll = self.payload.orientation.as_euler("ZYX", degrees=True)
        vel = self.payload.vel
        
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
        
        # Phase names for display
        phase_names = {
            0: "TAKEOFF",
            1: "HOVER-1",
            2: "TRANSLATE-E",
            3: "HOVER-2",
            4: "TRANSLATE-N",
            5: "HOVER-3",
            6: "LANDING",
            7: "COMPLETE"
        }
        phase_name = phase_names.get(self.planner.phase, "UNKNOWN")
        
        # Calculate reference errors
        ref_pos = self.planner.get_reference_position()
        pos_err = np.linalg.norm(pos - ref_pos)
        
        # Live telemetry display
        print(
            f"[{t:6.2f}s] Phase: {phase_name:12s} | "
            f"Pos: N={pos[1]:7.3f} E={pos[0]:7.3f} D={-pos[2]:7.3f} | "
            f"Vel: {np.linalg.norm(vel):6.3f} m/s | "
            f"Att: Y={yaw:6.1f}° P={pitch:6.1f}° R={roll:6.1f}° | "
            f"Err: {pos_err:6.4f} m",
            end="\r",
        )
