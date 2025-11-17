"""
Configuration constants for the drone-payload simulation.

This module centralizes all simulation parameters including physics constants,
controller gains, noise parameters, and feature toggles for the quadcopter
cooperative payload transport system.
"""

import numpy as np

# ============================================================================
# FEATURE SWITCHES
# ============================================================================
# Enable/disable various simulation features for testing and debugging

# Noise switches
ENABLE_DRONE_MASS_VARIANCE = True  # Add random variance to drone masses
ENABLE_DRONE_THRUST_NOISE = True  # Add noise to drone thrust outputs
ENABLE_ATTACHMENT_ERROR = True  # Add error to attachment point positions
ENABLE_WIND = True  # Enable wind disturbances

# Controller switches
ENABLE_ATTITUDE_CONTROLLER = True  # Enable geometric attitude control
ENABLE_POSITION_CONTROLLER = True  # Enable PID position control

# Profiling
ENABLE_PROFILING = True  # Enable cProfile performance profiling

# ============================================================================
# SIMULATION TIMING
# ============================================================================
TIME_STEP = 0.01  # Simulation time step in seconds
SIM_DURATION = 45  # Total simulation duration in seconds

# ============================================================================
# SIMULATION PERFORMANCE
# ============================================================================
TIME_SCALE_FACTOR = 0.7  # Real-time scaling factor for simulation speed

# ============================================================================
# PHYSICS
# ============================================================================
G_ACCELERATION = -9.80665  # Gravity acceleration (m/s²)
AIR_DENSITY = 1.225  # Air density at sea level (kg/m³)

# ============================================================================
# PAYLOAD (CUBE) PARAMETERS
# ============================================================================
CUBE_MASS = 10  # kg
CUBE_SIZE = 0.25  # meters

# ============================================================================
# DRONE PARAMETERS (base values)
# DJI Matrice 300 RTK specs used as reference
# ============================================================================
DRONE_MASS = 6.3  # kg (base mass, subject to random variation)
DRONE_MAX_THRUST = 9 * -G_ACCELERATION / 0.7  # N
DRONE_TIME_CONSTANT = 1e-7  # Motor response time constant (seconds)

# Noise parameters
DRONE_MASS_VARIANCE_AMOUNT = 0.02  # Mass variation coefficient (2%)
DRONE_THRUST_NOISE_AMOUNT = 0.05  # 5% thrust magnitude noise

# Conditional noise values (set to 0 if disabled)
DRONE_MASS_VARIANCE = DRONE_MASS_VARIANCE_AMOUNT if ENABLE_DRONE_MASS_VARIANCE else 0.0
DRONE_THRUST_NOISE = DRONE_THRUST_NOISE_AMOUNT if ENABLE_DRONE_THRUST_NOISE else 0.0

# ============================================================================
# ATTACHMENT ERROR
# ============================================================================
ATTACHMENT_ERROR_SEED = 0  # Random seed for attachment position error
ATTACHMENT_ERROR_SCALE_AMOUNT = 0.01  # Error scale relative to payload size (1%)

# Conditional attachment error (set to 0 if disabled)
ATTACHMENT_ERROR_SCALE = (
    ATTACHMENT_ERROR_SCALE_AMOUNT if ENABLE_ATTACHMENT_ERROR else 0.0
)

# ============================================================================
# WIND MODEL
# ============================================================================
WIND_SEED = 0  # Random seed for wind turbulence
WIND_BASELINE_AMOUNT = np.array([0.5, 0.0, 0.0])  # Baseline wind velocity (m/s)
WIND_GUST_SIGMA_AMOUNT = 0.3  # Gaussian random walk standard deviation (m/s/√s)
WIND_DRAG_COEFF = 1.0  # Drag coefficient for payload

# Conditional wind parameters (set to 0 if disabled)
WIND_BASELINE = WIND_BASELINE_AMOUNT if ENABLE_WIND else np.zeros(3)
WIND_GUST_SIGMA = WIND_GUST_SIGMA_AMOUNT if ENABLE_WIND else 0.0

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
KR_AMOUNT = 4.0  # Attitude error gain
KW_AMOUNT = 4.0  # Angular velocity error gain

# Position controller gains (Ziegler-Nichols tuned)
KU_XY = 0.08 * DRONE_MAX_THRUST  # Ultimate gain for X/Y axes
TU_XY = 3.5  # Ultimate period for X/Y axes
KU_Z = 0.12 * DRONE_MAX_THRUST  # Ultimate gain for Z axis
TU_Z = 3.5  # Ultimate period for Z axis

# PID gain amounts (computed from Ziegler-Nichols parameters)
KP_AMOUNT = np.diag([0.8 * KU_XY, 0.8 * KU_XY, 0.8 * KU_Z])
KI_POS_AMOUNT = np.diag([0 * KU_XY / TU_XY, 0 * KU_XY / TU_XY, 0 * KU_Z / TU_Z])
KD_AMOUNT = np.diag([0.1 * KU_XY * TU_XY, 0.1 * KU_XY * TU_XY, 0.1 * KU_Z * TU_Z])

# Conditional controller gains (set to 0 if disabled)
KR = KR_AMOUNT * DRONE_MAX_THRUST if ENABLE_ATTITUDE_CONTROLLER else 0.0
KW = KW_AMOUNT * DRONE_MAX_THRUST if ENABLE_ATTITUDE_CONTROLLER else 0.0
KP = KP_AMOUNT if ENABLE_POSITION_CONTROLLER else np.zeros((3, 3))
KI_POS = KI_POS_AMOUNT if ENABLE_POSITION_CONTROLLER else np.zeros((3, 3))
KD = KD_AMOUNT if ENABLE_POSITION_CONTROLLER else np.zeros((3, 3))

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

DRONE_TELEMETRY_FIELDS = [
    "irl_time",
    "pos",
    "command",
    "thrust",
    "feedforward",
    "feedback",
    "attitude",
]

PAYLOAD_TELEMETRY_FIELDS = [
    "irl_time",
    "pos",
    "rot",
    "acc",
    "vel",
    "ang_acc",
    "ang_vel",
    "wind",
]

MMAP_SIZE = 4096  # Initial memory-mapped file size (bytes)
POLLING_RATE = 500  # Hz - telemetry polling frequency
