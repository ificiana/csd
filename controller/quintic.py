"""
Quintic polynomial trajectory generation.

Theory:
    Quintic (5th-order) polynomial trajectories provide smooth motion profiles
    with continuous position, velocity, and acceleration. For a trajectory from
    rest to rest over distance d in time T:

        s(t) = A t^3 + B t^4 + C t^5

    Boundary conditions:
        s(0) = 0,  s'(0) = 0,  s''(0) = 0
        s(T) = d,  s'(T) = 0,  s''(T) = 0

    These 6 constraints uniquely determine coefficients A, B, C.

    The trajectory is parameterized by a coefficient c that relates to the
    maximum acceleration available:
        T = sqrt(3d / a_max) * scaling_factor

    The derivatives are:
        v(t) = s'(t) = 3A t^2 + 4B t^3 + 5C t^4
        a(t) = s''(t) = 6A t + 12B t^2 + 20C t^3
"""


def position(coeff: float, target_distance: float, time: float) -> float:
    """
    Quintic position trajectory ensuring smooth acceleration profile.

    Args:
        coeff: Trajectory coefficient related to maximum acceleration.
        target_distance: Total distance to travel.
        time: Current time along trajectory.

    Returns:
        Position at given time.
    """
    if target_distance <= 0 or coeff == 0:
        return 0.0
    A = 10 * coeff**3 / target_distance**2
    B = -15 * coeff**4 / target_distance**3
    C = 6 * coeff**5 / target_distance**4
    return A * time**3 + B * time**4 + C * time**5


def velocity(coeff: float, target_distance: float, time: float) -> float:
    """
    Velocity trajectory (derivative of position polynomial).

    Args:
        coeff: Trajectory coefficient.
        target_distance: Total distance to travel.
        time: Current time along trajectory.

    Returns:
        Velocity at given time.
    """
    if target_distance <= 0 or coeff == 0:
        return 0.0
    A = 30 * coeff**3 / target_distance**2
    B = -60 * coeff**4 / target_distance**3
    C = 30 * coeff**5 / target_distance**4
    return A * time**2 + B * time**3 + C * time**4


def acceleration(coeff: float, target_distance: float, time: float) -> float:
    """
    Acceleration trajectory (second derivative of position polynomial).

    Args:
        coeff: Trajectory coefficient.
        target_distance: Total distance to travel.
        time: Current time along trajectory.

    Returns:
        Acceleration at given time.
    """
    if target_distance <= 0 or coeff == 0:
        return 0.0
    A = 60 * coeff**3 / target_distance**2
    B = -180 * coeff**4 / target_distance**3
    C = 120 * coeff**5 / target_distance**4
    return A * time + B * time**2 + C * time**3
