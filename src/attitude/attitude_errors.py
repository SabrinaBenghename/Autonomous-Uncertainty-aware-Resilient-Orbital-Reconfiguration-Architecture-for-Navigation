import numpy as np


# ============================================================
# AURORA
# Controlled attitude-error utilities
#
# Convention:
#
#     R_BI_true
#
# transforms BODY -> ECI.
#
# We create an estimated attitude:
#
#     R_BI_est = R_BI_true @ R_error
#
# where R_error is a controlled rotation expressed
# relative to the BODY axes.
# ============================================================


def normalize_axis(
    axis,
    tolerance=1e-15
):
    """
    Normalise un axe de rotation.
    """

    axis = np.asarray(
        axis,
        dtype=float
    )


    norm = np.linalg.norm(
        axis
    )


    if norm < tolerance:

        raise ValueError(
            "L'axe de rotation ne peut pas etre nul."
        )


    return (
        axis
        / norm
    )


# ------------------------------------------------------------
# 1. MATRICE SKEW-SYMMETRIQUE
# ------------------------------------------------------------

def skew_symmetric(
    vector
):
    """
    Pour :

        v = [vx, vy, vz]

    retourne la matrice [v]_x telle que :

        [v]_x w = v x w
    """

    vector = np.asarray(
        vector,
        dtype=float
    )


    vx, vy, vz = (
        vector
    )


    return np.array([
        [
            0.0,
            -vz,
            vy
        ],
        [
            vz,
            0.0,
            -vx
        ],
        [
            -vy,
            vx,
            0.0
        ]
    ])


# ------------------------------------------------------------
# 2. ROTATION AXE-ANGLE
# ------------------------------------------------------------

def axis_angle_rotation_matrix(
    axis,
    angle_rad
):
    """
    Matrice de rotation Rodrigues.

    R =
        I cos(theta)
        + (1-cos(theta)) uu^T
        + sin(theta) [u]_x
    """

    unit_axis = normalize_axis(
        axis
    )


    angle_rad = float(
        angle_rad
    )


    cosine = np.cos(
        angle_rad
    )


    sine = np.sin(
        angle_rad
    )


    outer_product = np.outer(
        unit_axis,
        unit_axis
    )


    cross_matrix = skew_symmetric(
        unit_axis
    )


    rotation_matrix = (
        cosine
        * np.eye(
            3
        )
        +
        (
            1.0
            - cosine
        )
        * outer_product
        +
        sine
        * cross_matrix
    )


    return rotation_matrix


# ------------------------------------------------------------
# 3. PERTURBATION D'ATTITUDE
# ------------------------------------------------------------

def apply_body_fixed_attitude_error(
    true_rotation_body_to_eci,
    error_angle_rad,
    error_axis_body
):
    """
    Construit une attitude estimee erronee :

        R_BI_est =
            R_BI_true
            @ R_error

    L'erreur est donc appliquee dans les axes BODY.
    """

    true_rotation_body_to_eci = np.asarray(
        true_rotation_body_to_eci,
        dtype=float
    )


    error_rotation = (
        axis_angle_rotation_matrix(
            axis=
                error_axis_body,

            angle_rad=
                error_angle_rad
        )
    )


    estimated_rotation_body_to_eci = (
        true_rotation_body_to_eci
        @ error_rotation
    )


    return (
        estimated_rotation_body_to_eci
    )


# ------------------------------------------------------------
# 4. ANGLE ENTRE DEUX ATTITUDES
# ------------------------------------------------------------

def rotation_error_angle(
    true_rotation_body_to_eci,
    estimated_rotation_body_to_eci
):
    """
    Calcule l'angle principal entre deux rotations.
    """

    true_rotation_body_to_eci = np.asarray(
        true_rotation_body_to_eci,
        dtype=float
    )


    estimated_rotation_body_to_eci = np.asarray(
        estimated_rotation_body_to_eci,
        dtype=float
    )


    relative_rotation = (
        true_rotation_body_to_eci.T
        @ estimated_rotation_body_to_eci
    )


    cosine_angle = (
        (
            np.trace(
                relative_rotation
            )
            - 1.0
        )
        / 2.0
    )


    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )


    return np.arccos(
        cosine_angle
    )