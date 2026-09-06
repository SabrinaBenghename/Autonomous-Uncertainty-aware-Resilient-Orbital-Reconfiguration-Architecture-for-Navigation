import numpy as np

from src.dynamics.orbit import (
    MU_EARTH,
    R_EARTH,
    orbital_period,
    propagate_orbit
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)


# ============================================================
# AURORA
# Phase 4 - Constellation GNSS simplifiée
# ============================================================


# ------------------------------------------------------------
# 1. PARAMETRES GPS SIMPLIFIES
# ------------------------------------------------------------

GPS_ALTITUDE = 20_200_000.0       # [m]

GPS_SEMI_MAJOR_AXIS = (
    R_EARTH + GPS_ALTITUDE
)

GPS_ECCENTRICITY = 0.0

GPS_INCLINATION = np.deg2rad(
    55.0
)

GPS_NUMBER_OF_PLANES = 6

GPS_SATELLITES_PER_PLANE = 4

GPS_TOTAL_SATELLITES = (
    GPS_NUMBER_OF_PLANES
    * GPS_SATELLITES_PER_PLANE
)


# ------------------------------------------------------------
# 2. CREATION DE LA CONSTELLATION
# ------------------------------------------------------------

def generate_simplified_gps_constellation():
    """
    Génère les éléments orbitaux d'une
    constellation GPS simplifiée.

    Returns
    -------
    list of dict
        Liste contenant les paramètres
        orbitaux de chaque satellite GPS.
    """

    constellation = []

    satellite_id = 1

    for plane_index in range(
        GPS_NUMBER_OF_PLANES
    ):

        # Répartition régulière des plans
        # autour de la Terre
        raan = (
            2.0
            * np.pi
            * plane_index
            / GPS_NUMBER_OF_PLANES
        )

        for satellite_index in range(
            GPS_SATELLITES_PER_PLANE
        ):

            # Répartition régulière des satellites
            # dans chaque plan orbital
            true_anomaly = (
                2.0
                * np.pi
                * satellite_index
                / GPS_SATELLITES_PER_PLANE
            )

            # Petit déphasage entre les plans
            true_anomaly += (
                plane_index
                * np.pi
                / GPS_NUMBER_OF_PLANES
            )

            satellite = {
                "id": satellite_id,
                "semi_major_axis":
                    GPS_SEMI_MAJOR_AXIS,
                "eccentricity":
                    GPS_ECCENTRICITY,
                "inclination":
                    GPS_INCLINATION,
                "raan":
                    raan,
                "argument_of_periapsis":
                    0.0,
                "true_anomaly":
                    true_anomaly
            }

            constellation.append(
                satellite
            )

            satellite_id += 1

    return constellation


# ------------------------------------------------------------
# 3. ETAT CARTESIEN INITIAL D'UN SATELLITE GNSS
# ------------------------------------------------------------

def gnss_initial_state(
    satellite
):
    """
    Convertit les éléments orbitaux d'un
    satellite GNSS en état ECI initial.
    """

    position, velocity = (
        keplerian_to_cartesian(
            satellite[
                "semi_major_axis"
            ],
            satellite[
                "eccentricity"
            ],
            satellite[
                "inclination"
            ],
            satellite[
                "raan"
            ],
            satellite[
                "argument_of_periapsis"
            ],
            satellite[
                "true_anomaly"
            ]
        )
    )

    return np.concatenate(
        (
            position,
            velocity
        )
    )


# ------------------------------------------------------------
# 4. PROPAGATION DE LA CONSTELLATION
# ------------------------------------------------------------

def propagate_gnss_constellation(
    constellation,
    duration,
    number_of_points
):
    """
    Propage tous les satellites GNSS.

    Parameters
    ----------
    constellation : list
        Liste des satellites.

    duration : float
        Durée de simulation [s].

    number_of_points : int
        Nombre de points temporels.

    Returns
    -------
    list
        Liste des solutions orbitales.
    """

    solutions = []

    for satellite in constellation:

        initial_state = (
            gnss_initial_state(
                satellite
            )
        )

        solution = propagate_orbit(
            initial_state=initial_state,
            duration=duration,
            number_of_points=number_of_points
        )

        solutions.append(
            solution
        )

    return solutions


# ------------------------------------------------------------
# 5. PERIODE GPS APPROXIMATIVE
# ------------------------------------------------------------

GPS_ORBITAL_PERIOD = orbital_period(
    GPS_SEMI_MAJOR_AXIS
)