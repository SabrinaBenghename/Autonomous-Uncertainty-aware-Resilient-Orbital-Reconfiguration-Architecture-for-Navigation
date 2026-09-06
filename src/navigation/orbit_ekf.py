import numpy as np

from src.dynamics.orbit import MU_EARTH


# ============================================================
# AURORA
# Phase 7 - Extended Kalman Filter orbital
#
# Etat :
#
# x = [rx, ry, rz, vx, vy, vz]^T
#
# ============================================================


# ------------------------------------------------------------
# 1. DYNAMIQUE CONTINUE
# ------------------------------------------------------------

def orbital_state_derivative(state):
    """
    Dynamique orbitale à deux corps.

    Parameters
    ----------
    state : ndarray
        [rx, ry, rz, vx, vy, vz]

    Returns
    -------
    ndarray
        Dérivée de l'état.
    """

    position = state[0:3]
    velocity = state[3:6]

    radius = np.linalg.norm(
        position
    )

    acceleration = (
        -MU_EARTH
        * position
        / radius**3
    )

    derivative = np.zeros(
        6
    )

    derivative[0:3] = velocity
    derivative[3:6] = acceleration

    return derivative


# ------------------------------------------------------------
# 2. JACOBIENNE DE LA DYNAMIQUE
# ------------------------------------------------------------

def orbital_dynamics_jacobian(state):
    """
    Jacobienne continue F de la dynamique
    orbitale à deux corps.

    Returns
    -------
    ndarray
        Matrice F 6x6.
    """

    position = state[0:3]

    radius = np.linalg.norm(
        position
    )

    identity_3 = np.eye(
        3
    )

    outer_product = np.outer(
        position,
        position
    )

    gravity_gradient = (
        -MU_EARTH
        * (
            identity_3 / radius**3
            -
            3.0
            * outer_product / radius**5
        )
    )

    F = np.zeros(
        (
            6,
            6
        )
    )

    F[0:3, 3:6] = identity_3

    F[3:6, 0:3] = (
        gravity_gradient
    )

    return F


# ------------------------------------------------------------
# 3. PROPAGATION RK4 DE L'ETAT
# ------------------------------------------------------------

def rk4_propagate_state(
    state,
    dt
):
    """
    Propagation de l'état sur dt avec RK4.
    """

    k1 = orbital_state_derivative(
        state
    )

    k2 = orbital_state_derivative(
        state
        + 0.5 * dt * k1
    )

    k3 = orbital_state_derivative(
        state
        + 0.5 * dt * k2
    )

    k4 = orbital_state_derivative(
        state
        + dt * k3
    )

    return (
        state
        + dt
        / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )


# ------------------------------------------------------------
# 4. PREDICTION EKF
# ------------------------------------------------------------

def ekf_predict(
    state,
    covariance,
    dt,
    process_noise_covariance
):
    """
    Etape de prédiction de l'EKF.
    """

    # Propagation non linéaire de l'état
    predicted_state = (
        rk4_propagate_state(
            state,
            dt
        )
    )

    # Jacobienne autour de l'état courant
    F = orbital_dynamics_jacobian(
        state
    )

    # Approximation discrète :
    #
    # Phi ≈ I + F dt
    #
    state_transition_matrix = (
        np.eye(6)
        + F * dt
    )

    predicted_covariance = (
        state_transition_matrix
        @ covariance
        @ state_transition_matrix.T
        + process_noise_covariance
    )

    return (
        predicted_state,
        predicted_covariance
    )


# ------------------------------------------------------------
# 5. CORRECTION PAR POSITION GNSS
# ------------------------------------------------------------

def ekf_update_position(
    predicted_state,
    predicted_covariance,
    position_measurement,
    measurement_noise_covariance
):
    """
    Mise à jour EKF avec une mesure
    de position GNSS 3D.
    """

    # --------------------------------------------------------
    # Modèle de mesure :
    #
    # z = Hx + bruit
    #
    # Ici GNSS mesure directement r.
    # --------------------------------------------------------

    H = np.zeros(
        (
            3,
            6
        )
    )

    H[:, 0:3] = np.eye(
        3
    )

    predicted_measurement = (
        H
        @ predicted_state
    )

    innovation = (
        position_measurement
        - predicted_measurement
    )

    innovation_covariance = (
        H
        @ predicted_covariance
        @ H.T
        + measurement_noise_covariance
    )

    kalman_gain = (
        predicted_covariance
        @ H.T
        @ np.linalg.inv(
            innovation_covariance
        )
    )

    updated_state = (
        predicted_state
        + kalman_gain
        @ innovation
    )

    # --------------------------------------------------------
    # Joseph form
    # Plus robuste numériquement pour P
    # --------------------------------------------------------

    identity = np.eye(
        6
    )

    I_minus_KH = (
        identity
        - kalman_gain
        @ H
    )

    updated_covariance = (
        I_minus_KH
        @ predicted_covariance
        @ I_minus_KH.T
        +
        kalman_gain
        @ measurement_noise_covariance
        @ kalman_gain.T
    )

    return {
        "state":
            updated_state,

        "covariance":
            updated_covariance,

        "innovation":
            innovation,

        "innovation_covariance":
            innovation_covariance,

        "kalman_gain":
            kalman_gain
    }