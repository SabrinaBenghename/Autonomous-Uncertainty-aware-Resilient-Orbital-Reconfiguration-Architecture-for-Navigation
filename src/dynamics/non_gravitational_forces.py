import numpy as np

from scipy.integrate import solve_ivp

from src.dynamics.perturbations import (
    two_body_acceleration,
    j2_acceleration
)


# ============================================================
# AURORA
# Forces non gravitationnelles simplifiées
# ============================================================


# ------------------------------------------------------------
# 1. PERTURBATION TYPE TRAINEE
# ------------------------------------------------------------

def synthetic_drag_like_acceleration(
    time,
    position,
    velocity,
    base_acceleration=2.0e-5,
    modulation_amplitude=0.40,
    modulation_period=1800.0
):
    """
    Accélération non gravitationnelle synthétique,
    opposée à la vitesse orbitale.

    IMPORTANT :
    Ce n'est pas encore un modèle atmosphérique
    haute fidélité.

    Il s'agit d'une perturbation "drag-like"
    contrôlée pour tester la navigation.

    Parameters
    ----------
    time : float
        Temps [s].

    position : ndarray
        Position ECI [m].

    velocity : ndarray
        Vitesse ECI [m/s].

    base_acceleration : float
        Niveau moyen de perturbation [m/s^2].

    modulation_amplitude : float
        Modulation relative.

    modulation_period : float
        Période de modulation [s].

    Returns
    -------
    ndarray
        Accélération non gravitationnelle ECI [m/s^2].
    """

    del position

    speed = np.linalg.norm(
        velocity
    )

    if speed <= 0.0:

        return np.zeros(
            3
        )

    velocity_direction = (
        velocity
        / speed
    )

    modulation = (
        1.0
        +
        modulation_amplitude
        * np.sin(
            2.0
            * np.pi
            * time
            / modulation_period
        )
    )

    acceleration_magnitude = (
        base_acceleration
        * modulation
    )

    acceleration = (
        -acceleration_magnitude
        * velocity_direction
    )

    return acceleration


# ------------------------------------------------------------
# 2. DYNAMIQUE VRAIE
# ------------------------------------------------------------

def j2_and_drag_like_dynamics(
    time,
    state,
    base_acceleration=2.0e-5
):
    """
    Vérité dynamique :

        gravité centrale
        + J2
        + perturbation non gravitationnelle.
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

    j2_perturbation = (
        j2_acceleration(
            position
        )
    )

    non_gravitational_acceleration = (
        synthetic_drag_like_acceleration(
            time=
                time,

            position=
                position,

            velocity=
                velocity,

            base_acceleration=
                base_acceleration
        )
    )

    total_acceleration = (
        central_acceleration
        + j2_perturbation
        + non_gravitational_acceleration
    )

    return np.concatenate(
        (
            velocity,
            total_acceleration
        )
    )


# ------------------------------------------------------------
# 3. PROPAGATION DE LA VERITE
# ------------------------------------------------------------

def propagate_orbit_with_j2_and_drag_like(
    initial_state,
    duration,
    number_of_points,
    base_acceleration=2.0e-5
):
    """
    Propagation de la trajectoire vraie.
    """

    time_points = np.linspace(
        0.0,
        duration,
        number_of_points
    )

    solution = solve_ivp(
        fun=lambda t, x:
            j2_and_drag_like_dynamics(
                time=t,
                state=x,
                base_acceleration=
                    base_acceleration
            ),

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
            "Propagation J2 + perturbation échouée."
        )

    return solution