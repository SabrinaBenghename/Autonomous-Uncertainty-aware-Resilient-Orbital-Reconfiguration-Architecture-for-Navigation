import numpy as np


# ============================================================
# AURORA
# Phase 3 - Transformation des repères
# ECI <-> ECEF
# ============================================================


# ------------------------------------------------------------
# 1. CONSTANTE DE ROTATION TERRESTRE
# ------------------------------------------------------------

# Vitesse angulaire terrestre [rad/s]
OMEGA_EARTH = 7.2921150e-5


# ------------------------------------------------------------
# 2. MATRICE DE ROTATION AUTOUR DE Z
# ------------------------------------------------------------

def rotation_matrix_z(angle):
    """
    Matrice de rotation active autour de l'axe Z.

    Parameters
    ----------
    angle : float
        Angle de rotation [rad].

    Returns
    -------
    ndarray
        Matrice 3x3.
    """

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


# ------------------------------------------------------------
# 3. ANGLE DE ROTATION DE LA TERRE
# ------------------------------------------------------------

def earth_rotation_angle(
    time,
    initial_angle=0.0
):
    """
    Calcule l'angle de rotation terrestre simplifié.

    theta(t) = theta0 + omega_E * t

    Parameters
    ----------
    time : float
        Temps écoulé depuis l'époque initiale [s].

    initial_angle : float
        Angle terrestre à t = 0 [rad].

    Returns
    -------
    float
        Angle de rotation terrestre [rad].
    """

    return (
        initial_angle
        + OMEGA_EARTH * time
    )


# ------------------------------------------------------------
# 4. POSITION ECI -> ECEF
# ------------------------------------------------------------

def eci_to_ecef_position(
    position_eci,
    time,
    initial_angle=0.0
):
    """
    Transforme une position ECI en position ECEF.

    Parameters
    ----------
    position_eci : ndarray
        Position dans le repère ECI [m].

    time : float
        Temps écoulé [s].

    initial_angle : float
        Angle initial de rotation terrestre [rad].

    Returns
    -------
    ndarray
        Position dans le repère ECEF [m].
    """

    theta = earth_rotation_angle(
        time,
        initial_angle
    )

    transformation = rotation_matrix_z(
        -theta
    )

    return (
        transformation
        @ position_eci
    )


# ------------------------------------------------------------
# 5. POSITION ECEF -> ECI
# ------------------------------------------------------------

def ecef_to_eci_position(
    position_ecef,
    time,
    initial_angle=0.0
):
    """
    Transforme une position ECEF en position ECI.
    """

    theta = earth_rotation_angle(
        time,
        initial_angle
    )

    transformation = rotation_matrix_z(
        theta
    )

    return (
        transformation
        @ position_ecef
    )


# ------------------------------------------------------------
# 6. VITESSE ECI -> ECEF
# ------------------------------------------------------------

def eci_to_ecef_velocity(
    position_eci,
    velocity_eci,
    time,
    initial_angle=0.0
):
    """
    Transforme la vitesse ECI en vitesse ECEF.

    La rotation terrestre doit être prise en compte.

    v_ECEF =
        R * (v_ECI - omega x r_ECI)

    Parameters
    ----------
    position_eci : ndarray
        Position ECI [m].

    velocity_eci : ndarray
        Vitesse ECI [m/s].

    time : float
        Temps [s].

    initial_angle : float
        Angle initial terrestre [rad].

    Returns
    -------
    ndarray
        Vitesse ECEF [m/s].
    """

    theta = earth_rotation_angle(
        time,
        initial_angle
    )

    transformation = rotation_matrix_z(
        -theta
    )

    omega_vector = np.array([
        0.0,
        0.0,
        OMEGA_EARTH
    ])

    rotational_velocity = np.cross(
        omega_vector,
        position_eci
    )

    return (
        transformation
        @ (
            velocity_eci
            - rotational_velocity
        )
    )


# ------------------------------------------------------------
# 7. VITESSE ECEF -> ECI
# ------------------------------------------------------------

def ecef_to_eci_velocity(
    position_ecef,
    velocity_ecef,
    time,
    initial_angle=0.0
):
    """
    Transforme la vitesse ECEF en vitesse ECI.
    """

    theta = earth_rotation_angle(
        time,
        initial_angle
    )

    transformation = rotation_matrix_z(
        theta
    )

    position_eci = (
        transformation
        @ position_ecef
    )

    velocity_rotated = (
        transformation
        @ velocity_ecef
    )

    omega_vector = np.array([
        0.0,
        0.0,
        OMEGA_EARTH
    ])

    return (
        velocity_rotated
        + np.cross(
            omega_vector,
            position_eci
        )
    )