import numpy as np

from scipy.integrate import solve_ivp

from src.dynamics.orbit import (
    MU_EARTH,
    R_EARTH
)


# ============================================================
# AURORA
# Phase 7 - Perturbations orbitales
# Modèle J2
# ============================================================


# ------------------------------------------------------------
# 1. COEFFICIENT J2 TERRESTRE
# ------------------------------------------------------------

J2_EARTH = 1.08262668e-3


# ------------------------------------------------------------
# 2. ACCELERATION DEUX CORPS
# ------------------------------------------------------------

def two_body_acceleration(
    position
):
    """
    Accélération gravitationnelle centrale.

    Parameters
    ----------
    position : ndarray
        Position ECI [m].

    Returns
    -------
    ndarray
        Accélération [m/s^2].
    """

    radius = np.linalg.norm(
        position
    )

    return (
        -MU_EARTH
        * position
        / radius**3
    )


# ------------------------------------------------------------
# 3. ACCELERATION J2
# ------------------------------------------------------------

def j2_acceleration(
    position
):
    """
    Calcule la perturbation d'accélération
    due au coefficient J2 terrestre.

    Parameters
    ----------
    position : ndarray
        Position ECI [m].

    Returns
    -------
    ndarray
        Accélération perturbatrice J2 [m/s^2].
    """

    x, y, z = position

    radius = np.linalg.norm(
        position
    )

    radius_squared = (
        radius**2
    )

    z_squared = (
        z**2
    )

    common_factor = (
        1.5
        * J2_EARTH
        * MU_EARTH
        * R_EARTH**2
        / radius**5
    )

    common_xy_term = (
        5.0
        * z_squared
        / radius_squared
        - 1.0
    )

    z_term = (
        5.0
        * z_squared
        / radius_squared
        - 3.0
    )

    acceleration_x = (
        common_factor
        * x
        * common_xy_term
    )

    acceleration_y = (
        common_factor
        * y
        * common_xy_term
    )

    acceleration_z = (
        common_factor
        * z
        * z_term
    )

    return np.array([
        acceleration_x,
        acceleration_y,
        acceleration_z
    ])


# ------------------------------------------------------------
# 4. DYNAMIQUE DEUX CORPS + J2
# ------------------------------------------------------------

def j2_orbit_dynamics(
    time,
    state
):
    """
    Dynamique orbitale incluant J2.

    Etat :
        [rx, ry, rz, vx, vy, vz]
    """

    position = (
        state[0:3]
    )

    velocity = (
        state[3:6]
    )

    central_acceleration = (
        two_body_acceleration(
            position
        )
    )

    perturbation_acceleration = (
        j2_acceleration(
            position
        )
    )

    total_acceleration = (
        central_acceleration
        + perturbation_acceleration
    )

    return np.concatenate(
        (
            velocity,
            total_acceleration
        )
    )


# ------------------------------------------------------------
# 5. PROPAGATION AVEC J2
# ------------------------------------------------------------

def propagate_orbit_with_j2(
    initial_state,
    duration,
    number_of_points=5000
):
    """
    Propage une orbite avec gravité centrale
    + perturbation J2.
    """

    time_points = np.linspace(
        0.0,
        duration,
        number_of_points
    )

    solution = solve_ivp(
        fun=
            j2_orbit_dynamics,

        t_span=(
            0.0,
            duration
        ),

        y0=
            initial_state,

        t_eval=
            time_points,

        rtol=
            1e-9,

        atol=
            1e-9
    )

    if not solution.success:

        raise RuntimeError(
            "La propagation orbitale J2 a échoué."
        )

    return solution