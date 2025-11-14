"""
Configuration constants for the drone-payload simulation
"""

import numpy as np

# ============================================================================
# SIMULATION TIMING
# ============================================================================
TIME_STEP = 0.001  # Simulation time step in seconds
SIM_DURATION = 30  # Total simulation duration in seconds

# ============================================================================
# SIMULATION PERFORMANCE
# ============================================================================
TIME_SCALE_FACTOR = 0.7  # Real-time scaling factor for simulation speed

# ============================================================================
# PHYSICS
# ============================================================================
G_ACCELERATION = -9.80665  # Gravity acceleration (m/s²)

# ============================================================================
# PAYLOAD (CUBE) PARAMETERS
# ============================================================================
CUBE_MASS = 10  # kg
CUBE_SIZE = 1  # meters

# ============================================================================
# DRONE PARAMETERS
# ============================================================================
DRONE_MASS = 1  # kg (base mass, subject to random variation)
DRONE_MAX_THRUST = 50  # N
DRONE_MASS_VARIANCE = 0.0001  # Mass variation coefficient
DRONE_THRUST_NOISE = 0.01  # 1% thrust magnitude noise
DRONE_TIME_CONSTANT = 0.2  # Motor response time constant (seconds)

# ============================================================================
# ATTACHMENT ERROR
# ============================================================================
ATTACHMENT_ERROR_SEED = 0  # Random seed for attachment position error
ATTACHMENT_ERROR_SCALE = 0.0001  # Error scale relative to payload size

# ============================================================================
# TRAJECTORY WAYPOINTS
# ============================================================================
HEIGHT = 5  # Target hover height (meters)
EAST = 10  # Target eastward position (meters)
NORTH = 10  # Target northward position (meters)

# ============================================================================
# TRAJECTORY PHASE TIMING
# ============================================================================
HOVER_TIMES = [2.0, 2.0, 10.0]  # Hover durations at each waypoint (seconds)

# ============================================================================
# CONTROLLER GAINS
# ============================================================================
# Attitude control gains (Lee 2010 geometric controller)
KR = 100.0  # Attitude error gain
KW = 100.0  # Angular velocity error gain

# Position controller gains (Ziegler-Nichols tuned)
KU_XY = 4  # Ultimate gain for X/Y axes
TU_XY = 3.14  # Ultimate period for X/Y axes
KU_Z = 6  # Ultimate gain for Z axis
TU_Z = 2.56  # Ultimate period for Z axis

# PID gains (computed from Ziegler-Nichols parameters)
KP = np.diag([0.8 * KU_XY, 0.8 * KU_XY, 0.8 * KU_Z])
KI_POS = np.diag([0 * KU_XY / TU_XY, 0 * KU_XY / TU_XY, 0 * KU_Z / TU_Z])
KD = np.diag([0.1 * KU_XY * TU_XY, 0.1 * KU_XY * TU_XY, 0.1 * KU_Z * TU_Z])

# ============================================================================
# THRUST ALLOCATION
# ============================================================================
# Drone attachment points on payload (x, y signs)
# Pattern: (++), (+-), (-+), (--)
ATTACHMENT_POINTS = np.array(
    [
        [1, 1],  # drone 0: +x, +y
        [1, -1],  # drone 1: +x, -y
        [-1, 1],  # drone 2: -x, +y
        [-1, -1],  # drone 3: -x, -y
    ]
)

# 4x3 thrust mixing matrix for quadcopter configuration
# Drone positions: ++, -+, --, +-  (x, y relative to payload center)
MX = np.array(
    [
        [1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, -1],
    ]
)

# ============================================================================
# CONTROLLER THRUST LIMITS
# ============================================================================
MAX_TAKEOFF_FRACTION = 0.75  # Fraction of max thrust for takeoff
MAX_TRANSL_FRACTION = 0.33  # Fraction of residual thrust for translation

# ============================================================================
# NUMERICAL TOLERANCES
# ============================================================================
POSITION_TOLERANCE = 1e-6  # Tolerance for position comparisons

# ============================================================================
# TELEMETRY
# ============================================================================
SENSOR_CHANNELS = [
    "cube",
    "drone_0",
    "drone_1",
    "drone_2",
    "drone_3",
]

MMAP_SIZE = 4096  # Initial memory-mapped file size (bytes)
POLLING_RATE = 50  # Hz - telemetry polling frequency
