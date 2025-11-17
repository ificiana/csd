# Controller Module

Hierarchical control system for cooperative quadcopter payload transport.

## Module Structure

```
controller/
├── __init__.py      # Module exports
├── quintic.py       # Quintic polynomial trajectory generation
├── position.py      # PID position controller (outer loop)
├── attitude.py      # Geometric attitude controller (inner loop)
├── plan.py          # Multi-phase trajectory planner
└── thrust.py        # Main controller coordinator
```

## Components

### quintic.py
Quintic (5th-order) polynomial trajectory generation for smooth motion profiles.

**Functions:**
- `position(coeff, target_distance, time)` - Position trajectory s(t)
- `velocity(coeff, target_distance, time)` - Velocity trajectory s'(t)
- `acceleration(coeff, target_distance, time)` - Acceleration trajectory s''(t)

**Theory:**
Quintic polynomials provide C^2 continuous trajectories (continuous position,
velocity, and acceleration) with zero velocity and acceleration at endpoints.

### position.py
PID position controller for 3-DOF translational control.

**Class:** `PositionController`
- `compute_force(pos, vel, pos_ref, vel_ref)` - Computes desired force using PID law

**Theory:**
Classical PID control: F = m * (K_p * e_pos + K_d * e_vel + K_i * int(e_pos dt))

### attitude.py
Geometric attitude controller on SO(3) with thrust allocation.

**Class:** `AttitudeController`
- `compute_thrust_corrections()` - Computes per-drone thrust corrections

**Theory:**
Lee 2010 geometric control on the Lie group SO(3). Uses least-squares
allocation to map desired torque to individual drone forces.

### plan.py
Multi-phase trajectory state machine.

**Class:** `TrajectoryPlanner`
- `get_reference_position()` - Returns reference position for current phase
- `get_reference_velocity()` - Returns reference velocity for current phase
- `get_feedforward_acceleration(pos)` - Computes feedforward acceleration

**Phases:**
0. Takeoff - Vertical ascent
1. Hover - Stabilize at altitude
2. Translate East - Horizontal motion (x-axis)
3. Hover - Stabilize at waypoint
4. Translate North - Horizontal motion (y-axis)
5. Hover - Final position hold
6. Land - Controlled descent
7. Complete - End of trajectory

### thrust.py
Main controller coordinator integrating all components.

**Class:** `ThrustController`
- `update()` - Main control loop update

**Architecture:**
Coordinates trajectory planning, position control, attitude control, and
thrust allocation with saturation limits.

## Usage

```python
from controller import ThrustController

controller = ThrustController(drones=drones, payload=payload)

# In simulation loop:
controller.update()
```

## Control Flow

```
ThrustController.update()
├── TrajectoryPlanner.get_feedforward_acceleration(pos)
│   └── quintic.acceleration()
├── AttitudeController.compute_thrust_corrections()
│   └── [Least-squares thrust allocation]
├── PositionController.compute_force(pos, vel, pos_ref, vel_ref)
│   ├── TrajectoryPlanner.get_reference_position()
│   │   └── quintic.position()
│   └── TrajectoryPlanner.get_reference_velocity()
│       └── quintic.velocity()
└── [Combine and saturate thrusts]
```

## Mathematical Background

### Outer Loop (Position Control)
- **Type:** PID controller
- **DOF:** 3 (x, y, z translation)
- **Output:** Desired force vector F_des

### Inner Loop (Attitude Control)
- **Type:** Geometric controller on SO(3)
- **DOF:** 3 (roll, pitch, yaw rotation)
- **Output:** Per-drone thrust corrections

### Trajectory Generation
- **Type:** Quintic polynomials
- **Properties:** C^2 continuous, zero endpoint velocities
- **Boundary conditions:** 6 constraints determine unique solution

## References

1. Lee, T., Leok, M., & McClamroch, N. H. (2010). Geometric tracking control
   of a quadrotor UAV on SE(3). CDC 2010.

2. Ziegler, J. G., & Nichols, N. B. (1942). Optimum settings for automatic
   controllers. Transactions of the ASME, 64(11).
