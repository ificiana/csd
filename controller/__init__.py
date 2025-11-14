"""
Hierarchical thrust controller for cooperative payload transport.

This module orchestrates position control, attitude control, trajectory
planning, and thrust allocation for quadcopter swarm payload transport.

Architecture:
    - TrajectoryPlanner: Generates reference trajectories
    - PositionController: Outer loop (translational control)
    - AttitudeController: Inner loop (rotational control)
    - ThrustController: Coordinates all components
"""

from .attitude import AttitudeController
from .plan import TrajectoryPlanner
from .position import PositionController
from .thrust import ThrustController

__all__ = [
    "AttitudeController",
    "PositionController",
    "TrajectoryPlanner",
    "ThrustController",
]
