"""
Simulation clock management.

This module implements a discrete-time simulation clock with fixed time steps,
decoupled from real-world wall-clock time. The simulation time advances in
constant increments (TIME_STEP) to ensure numerical stability and reproducibility
of the physics integration.

Theory:
    For numerical integration of differential equations, fixed time-stepping
    provides consistent behavior across runs and simplifies the implementation
    of explicit integration schemes (e.g., forward Euler):

        x(t + dt) = x(t) + dx/dt * dt

    where dt = TIME_STEP is constant throughout the simulation.
"""

import time

from config import TIME_STEP

SIM_TIME = 0
START_TIME = time.time_ns()


def get_time():
    """Returns current simulation time in seconds."""
    return SIM_TIME


def time_tick():
    """Advances simulation time by one TIME_STEP and returns new time."""
    global SIM_TIME
    SIM_TIME += TIME_STEP
    return SIM_TIME
