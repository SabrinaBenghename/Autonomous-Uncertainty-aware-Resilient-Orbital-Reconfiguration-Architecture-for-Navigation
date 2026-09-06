import numpy as np

from src.dynamics.orbit import MU_EARTH


# ============================================================
# AURORA
# Phase 2 - Eléments orbitaux képlériens
# ============================================================


# ------------------------------------------------------------
# 1. MATRICES DE ROTATION
# ------------------------------------------------------------

def rotation_matrix_x(angle):
    """
    Matrice de rotation autour de l'axe X.

    Parameters
    ----------
    angle : float
        Angle [rad].

    Returns
    -------
    ndarray
        Matrice 3x3.
    """

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def rotation_matrix_z(angle):
    """
    Matrice de rotation autour de l'axe Z.

    Parameters
    ----------
    angle : float
        Angle [rad].

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
# 2. NORMALISATION ANGULAIRE
# ------------------------------------------------------------

def normalize_angle(angle):
    """
    Ramène un angle dans l'intervalle [0, 2*pi).
    """

    return angle % (2.0 * np.pi)


# ------------------------------------------------------------
# 3. KEPLERIEN -> CARTESIEN
# ------------------------------------------------------------

def keplerian_to_cartesian(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly
):
    """
    Convertit les éléments képlériens en
    position et vitesse dans le repère ECI.

    Parameters
    ----------
    semi_major_axis : float
        Demi-grand axe a [m].

    eccentricity : float
        Excentricité e [-].

    inclination : float
        Inclinaison i [rad].

    raan : float
        RAAN Omega [rad].

    argument_of_periapsis : float
        Argument du périgée omega [rad].

    true_anomaly : float
        Anomalie vraie nu [rad].

    Returns
    -------
    position_eci : ndarray
        Position ECI [m].

    velocity_eci : ndarray
        Vitesse ECI [m/s].
    """

    # Paramètre orbital
    p = (
        semi_major_axis
        * (1.0 - eccentricity**2)
    )

    # Rayon instantané
    radius = (
        p
        /
        (
            1.0
            + eccentricity
            * np.cos(true_anomaly)
        )
    )

    # Position dans le repère périfocal PQW
    position_pqw = np.array([
        radius * np.cos(true_anomaly),
        radius * np.sin(true_anomaly),
        0.0
    ])

    # Vitesse dans PQW
    velocity_pqw = (
        np.sqrt(MU_EARTH / p)
        * np.array([
            -np.sin(true_anomaly),
            eccentricity
            + np.cos(true_anomaly),
            0.0
        ])
    )

    # Transformation PQW -> ECI
    transformation_matrix = (
        rotation_matrix_z(raan)
        @ rotation_matrix_x(inclination)
        @ rotation_matrix_z(
            argument_of_periapsis
        )
    )

    position_eci = (
        transformation_matrix
        @ position_pqw
    )

    velocity_eci = (
        transformation_matrix
        @ velocity_pqw
    )

    return (
        position_eci,
        velocity_eci
    )


# ------------------------------------------------------------
# 4. CARTESIEN -> KEPLERIEN
# ------------------------------------------------------------

def cartesian_to_keplerian(
    position,
    velocity
):
    """
    Convertit une position et une vitesse ECI
    vers les éléments orbitaux képlériens.

    Parameters
    ----------
    position : ndarray
        Position ECI [m].

    velocity : ndarray
        Vitesse ECI [m/s].

    Returns
    -------
    dict
        Eléments orbitaux :
        a, e, i, raan, argument_of_periapsis,
        true_anomaly.
    """

    # --------------------------------------------------------
    # Normes
    # --------------------------------------------------------

    radius = np.linalg.norm(
        position
    )

    speed = np.linalg.norm(
        velocity
    )

    # --------------------------------------------------------
    # Moment cinétique spécifique
    # --------------------------------------------------------

    h_vector = np.cross(
        position,
        velocity
    )

    h = np.linalg.norm(
        h_vector
    )

    # --------------------------------------------------------
    # Vecteur nodal
    # --------------------------------------------------------

    k_vector = np.array([
        0.0,
        0.0,
        1.0
    ])

    n_vector = np.cross(
        k_vector,
        h_vector
    )

    n = np.linalg.norm(
        n_vector
    )

    # --------------------------------------------------------
    # Vecteur excentricité
    # --------------------------------------------------------

    eccentricity_vector = (
        np.cross(
            velocity,
            h_vector
        )
        / MU_EARTH
        -
        position / radius
    )

    eccentricity = np.linalg.norm(
        eccentricity_vector
    )

    # --------------------------------------------------------
    # Energie mécanique spécifique
    # --------------------------------------------------------

    specific_energy = (
        speed**2 / 2.0
        -
        MU_EARTH / radius
    )

    # --------------------------------------------------------
    # Demi-grand axe
    # --------------------------------------------------------

    semi_major_axis = (
        -MU_EARTH
        /
        (
            2.0
            * specific_energy
        )
    )

    # --------------------------------------------------------
    # Inclinaison
    # --------------------------------------------------------

    inclination = np.arccos(
        np.clip(
            h_vector[2] / h,
            -1.0,
            1.0
        )
    )

    tolerance = 1e-12

    # --------------------------------------------------------
    # RAAN
    # --------------------------------------------------------

    if n > tolerance:

        raan = np.arctan2(
            n_vector[1],
            n_vector[0]
        )

        raan = normalize_angle(
            raan
        )

    else:

        raan = 0.0

    # --------------------------------------------------------
    # Argument du périgée
    # --------------------------------------------------------

    if (
        n > tolerance
        and eccentricity > tolerance
    ):

        cos_argument = (
            np.dot(
                n_vector,
                eccentricity_vector
            )
            /
            (
                n
                * eccentricity
            )
        )

        sin_argument = (
            np.dot(
                np.cross(
                    n_vector,
                    eccentricity_vector
                ),
                h_vector
            )
            /
            (
                n
                * eccentricity
                * h
            )
        )

        argument_of_periapsis = np.arctan2(
            sin_argument,
            cos_argument
        )

        argument_of_periapsis = normalize_angle(
            argument_of_periapsis
        )

    else:

        argument_of_periapsis = 0.0

    # --------------------------------------------------------
    # Anomalie vraie
    # --------------------------------------------------------

    if eccentricity > tolerance:

        cos_true_anomaly = (
            np.dot(
                eccentricity_vector,
                position
            )
            /
            (
                eccentricity
                * radius
            )
        )

        sin_true_anomaly = (
            np.dot(
                np.cross(
                    eccentricity_vector,
                    position
                ),
                h_vector
            )
            /
            (
                eccentricity
                * radius
                * h
            )
        )

        true_anomaly = np.arctan2(
            sin_true_anomaly,
            cos_true_anomaly
        )

        true_anomaly = normalize_angle(
            true_anomaly
        )

    else:

        true_anomaly = 0.0

    return {
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination": inclination,
        "raan": raan,
        "argument_of_periapsis":
            argument_of_periapsis,
        "true_anomaly": true_anomaly
    }