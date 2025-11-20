# Hierarchical Control of Cooperative Quadrotor Payload Transport

A 6-DOF rigid body dynamics simulation for cooperative payload transport using multiple quadrotor drones with geometric attitude control.

**Term Project for ME5780 Control Systems Design**  
Indian Institute of Technology Hyderabad  
November 2025

## Authors

- Arkaprabha Sinha Roy (AI25MTECH12001)
- Rehbar Khan (ME25MTECH11015)
- S. Chinnikrishna Yadav (ME25MTECH12005)
- Vikram Kumar Anand (ME25MTECH14011)

## Overview

This project implements a high-fidelity simulation of four quadrotor drones cooperatively transporting a cubic payload. The system features:

- **6-DOF Rigid Body Dynamics**: Full Newton-Euler equations for payload motion on SO(3)
- **Hierarchical Control Architecture**: 
  - Outer-loop PID position controller
  - Inner-loop geometric attitude controller with least-squares thrust allocation
- **Quintic Polynomial Trajectories**: Smooth motion profiles with continuous acceleration
- **Realistic Actuator Modeling**: First-order motor dynamics with configurable time constants
- **Comprehensive Noise Injection**: 
  - Drone mass variance (2%)
  - Thrust noise (5%)
  - Attachment point errors (1%)
  - Wind disturbances with baseline + Brownian gusts
- **Multi-phase Mission Planning**: Takeoff, translation, and landing maneuvers

## Key Features

### Control System
- **Geometric Attitude Control on SO(3)**: Singularity-free orientation tracking using rotation matrices
- **PID Position Control**: Configurable gains tuned via Ziegler-Nichols method
- **Thrust Allocation**: Least-squares optimization to distribute desired wrench among four drones
- **Actuator Dynamics**: First-order low-pass filter modeling motor response lag

### Simulation Capabilities
- Real-time visualization support
- Comprehensive telemetry logging (JSON format)
- Configurable noise and disturbances
- Profiling support for performance analysis
- Feature toggles for testing individual components

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Trajectory Generator                     │
│              (Quintic Polynomial Planner)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Position Controller (PID)                      │
│         Tracks reference position, velocity, accel          │
└────────────────────┬────────────────────────────────────────┘
                     │ Desired Total Force
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Attitude Controller (Geometric)                   │
│      Computes thrust direction on SO(3) manifold            │
└────────────────────┬────────────────────────────────────────┘
                     │ Desired Wrench
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Thrust Allocation                              │
│    Least-squares optimization over 4 drone thrusts          │
└────────────────────┬────────────────────────────────────────┘
                     │ Individual Drone Commands
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Actuator Dynamics (1st order)                     │
│              T_actual = T_cmd / (τs + 1)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Rigid Body Dynamics (6-DOF)                        │
│   Newton-Euler equations with ground contact detection      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements
- Python ≥ 3.13
- Dependencies managed via `uv` or `pip`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd csd

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

## Usage

### Running the Simulation

```bash
python main.py
```

The simulation will:
1. Initialize the payload and four drones
2. Execute a multi-phase mission:
   - Takeoff from ground to 2.0m
   - Translate 5.0m in x-direction
   - Land back to ground
3. Generate telemetry data in `data/` directory
4. Display real-time visualization (if enabled)

### Configuration

All simulation parameters are centralized in `config.py`:

```python
# Feature toggles
ENABLE_DRONE_MASS_VARIANCE = True
ENABLE_DRONE_THRUST_NOISE = True
ENABLE_ATTACHMENT_ERROR = True
ENABLE_WIND = True
ENABLE_ATTITUDE_CONTROLLER = True
ENABLE_POSITION_CONTROLLER = True

# Simulation timing
TIME_STEP = 0.01  # seconds
SIM_DURATION = 45  # seconds

# Physical parameters
CUBE_MASS = 10  # kg
CUBE_SIZE = 0.25  # meters
DRONE_MASS = 6.3  # kg (per drone)
DRONE_MAX_THRUST = 126  # N (total)

# Noise parameters
DRONE_MASS_VARIANCE_AMOUNT = 0.02  # 2%
DRONE_THRUST_NOISE_AMOUNT = 0.05  # 5%
ATTACHMENT_ERROR_SCALE_AMOUNT = 0.01  # 1%
```

### Controller Tuning

PID gains can be adjusted in `config.py` or via the tuning utility:

```bash
python tune.py
```

Default gains (Ziegler-Nichols tuned):
- **Position (XY)**: Kp = 10.08, Ki = 0, Kd = 4.41
- **Position (Z)**: Kp = 15.12, Ki = 0, Kd = 6.62
- **Attitude**: KR = 504, Kω = 504

## Project Structure

```
csd/
├── main.py              # Main simulation entry point
├── config.py            # Configuration parameters
├── entity.py            # Base entity class
├── drone.py             # Drone model with actuator dynamics
├── receiver.py          # Payload rigid body dynamics
├── clock.py             # Simulation timing
├── wind.py              # Wind disturbance model
├── tel.py               # Telemetry logging
├── utils.py             # Utility functions
├── tune.py              # Controller tuning utilities
├── controller/          # Control algorithms
│   ├── position.py      # PID position controller
│   ├── attitude.py      # Geometric attitude controller
│   └── trajectory.py    # Quintic polynomial planner
├── viz/                 # Visualization tools
├── data/                # Telemetry output
├── expts/               # Experimental results
├── report.tex           # Detailed technical report
└── pyproject.toml       # Project dependencies
```

## Technical Details

See `report.tex` for detailed experimental results and analysis.
