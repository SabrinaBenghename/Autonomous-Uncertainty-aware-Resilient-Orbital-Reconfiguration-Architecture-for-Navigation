import numpy as np

from src.dynamics.orbit import (
    MU_EARTH,
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)


# ============================================================
# AURORA
# Interface GNSS -> EKF
#
# Pipeline :
#
# constellation simplifiee
#       ↓
# visibilite
#       ↓
# pseudodistances
#       ↓
# moindres carres
#       ↓
# position GNSS + covariance
# ============================================================


C_LIGHT = 299_792_458.0


# ------------------------------------------------------------
# 1. CONSTELLATION GPS SIMPLIFIEE
# ------------------------------------------------------------

GPS_ALTITUDE = (
    20_200_000.0
)

GPS_SEMI_MAJOR_AXIS = (
    R_EARTH
    + GPS_ALTITUDE
)

GPS_INCLINATION = np.deg2rad(
    55.0
)

GPS_NUMBER_OF_PLANES = (
    6
)

GPS_SATELLITES_PER_PLANE = (
    4
)

GPS_TOTAL_SATELLITES = (
    GPS_NUMBER_OF_PLANES
    * GPS_SATELLITES_PER_PLANE
)


# ------------------------------------------------------------
# 2. NOMS DES SATELLITES
# ------------------------------------------------------------

def get_gps_satellite_ids():

    return [
        f"GPS{index:02d}"
        for index in range(
            1,
            GPS_TOTAL_SATELLITES + 1
        )
    ]


# ------------------------------------------------------------
# 3. POSITIONS GPS A UNE EPOQUE
# ------------------------------------------------------------

def generate_simplified_gps_positions(
    time_seconds
):
    """
    Genere les positions ECI des 24 satellites
    GPS simplifies.

    Hypotheses :
        - orbites circulaires
        - altitude 20 200 km
        - inclinaison 55 deg
        - 6 plans
        - 4 satellites par plan
        - propagation Keplerienne circulaire
    """

    mean_motion = np.sqrt(
        MU_EARTH
        / GPS_SEMI_MAJOR_AXIS**3
    )


    positions = []

    satellite_ids = []


    satellite_counter = (
        1
    )


    for plane_index in range(
        GPS_NUMBER_OF_PLANES
    ):

        raan = (
            2.0
            * np.pi
            * plane_index
            / GPS_NUMBER_OF_PLANES
        )


        for satellite_index in range(
            GPS_SATELLITES_PER_PLANE
        ):

            initial_true_anomaly = (
                2.0
                * np.pi
                * satellite_index
                / GPS_SATELLITES_PER_PLANE

                +
                plane_index
                * np.pi
                / 6.0
            )


            true_anomaly = (
                initial_true_anomaly
                + mean_motion
                * time_seconds
            )


            true_anomaly = (
                true_anomaly
                % (
                    2.0
                    * np.pi
                )
            )


            (
                satellite_position,
                _
            ) = keplerian_to_cartesian(
                GPS_SEMI_MAJOR_AXIS,
                0.0,
                GPS_INCLINATION,
                raan,
                0.0,
                true_anomaly
            )


            positions.append(
                satellite_position
            )


            satellite_ids.append(
                f"GPS{satellite_counter:02d}"
            )


            satellite_counter += (
                1
            )


    return (
        np.asarray(
            positions
        ),

        satellite_ids
    )


# ------------------------------------------------------------
# 4. DISTANCE MINIMALE DU SEGMENT AU CENTRE TERRE
# ------------------------------------------------------------

def minimum_segment_radius(
    point_a,
    point_b
):
    """
    Distance minimale entre le centre de la Terre
    et le segment reliant deux points.
    """

    point_a = np.asarray(
        point_a,
        dtype=float
    )

    point_b = np.asarray(
        point_b,
        dtype=float
    )


    segment = (
        point_b
        - point_a
    )


    denominator = (
        segment
        @ segment
    )


    if denominator <= 0.0:

        return np.linalg.norm(
            point_a
        )


    parameter = (
        -point_a
        @ segment
        / denominator
    )


    parameter = np.clip(
        parameter,
        0.0,
        1.0
    )


    closest_point = (
        point_a
        + parameter
        * segment
    )


    return np.linalg.norm(
        closest_point
    )


# ------------------------------------------------------------
# 5. VISIBILITE GEOMETRIQUE
# ------------------------------------------------------------

def is_gnss_satellite_visible(
    receiver_position,
    satellite_position
):
    """
    Un satellite GNSS est considere visible
    si la Terre ne coupe pas la ligne de visee.
    """

    minimum_radius = (
        minimum_segment_radius(
            receiver_position,
            satellite_position
        )
    )


    return (
        minimum_radius
        >= R_EARTH
    )


# ------------------------------------------------------------
# 6. SATELLITES VISIBLES
# ------------------------------------------------------------

def select_visible_gnss_satellites(
    receiver_position,
    satellite_positions,
    satellite_ids
):

    visible_positions = []

    visible_ids = []


    for (
        satellite_position,
        satellite_id
    ) in zip(
        satellite_positions,
        satellite_ids
    ):

        if is_gnss_satellite_visible(
            receiver_position,
            satellite_position
        ):

            visible_positions.append(
                satellite_position
            )

            visible_ids.append(
                satellite_id
            )


    return (
        np.asarray(
            visible_positions
        ),

        visible_ids
    )


# ------------------------------------------------------------
# 7. SIMULATION DES PSEUDODISTANCES
# ------------------------------------------------------------

def simulate_pseudorange_measurements(
    receiver_position,
    satellite_positions,
    receiver_clock_bias_seconds,
    pseudorange_noise_std,
    rng
):
    """
    Modele :

        rho_i =
            ||r_sat_i - r_receiver||
            + c * dt_receiver
            + noise_i
    """

    receiver_position = np.asarray(
        receiver_position,
        dtype=float
    )


    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )


    geometric_ranges = np.linalg.norm(
        satellite_positions
        - receiver_position,
        axis=1
    )


    receiver_clock_bias_meters = (
        C_LIGHT
        * receiver_clock_bias_seconds
    )


    noise = rng.normal(
        loc=0.0,
        scale=pseudorange_noise_std,
        size=len(
            satellite_positions
        )
    )


    pseudoranges = (
        geometric_ranges
        + receiver_clock_bias_meters
        + noise
    )


    return {
        "pseudoranges":
            pseudoranges,

        "geometric_ranges":
            geometric_ranges,

        "clock_bias_meters":
            receiver_clock_bias_meters,

        "noise":
            noise
    }


# ------------------------------------------------------------
# 8. MATRICE GEOMETRIQUE GNSS
# ------------------------------------------------------------

def build_gnss_geometry_matrix(
    receiver_position,
    satellite_positions
):
    """
    H pour l'etat GNSS :

        [x, y, z, b]

    ou b est le biais d'horloge exprime en metres.
    """

    receiver_position = np.asarray(
        receiver_position,
        dtype=float
    )


    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )


    differences = (
        satellite_positions
        - receiver_position
    )


    ranges = np.linalg.norm(
        differences,
        axis=1
    )


    if np.any(
        ranges <= 0.0
    ):

        raise ValueError(
            "Distance GNSS invalide."
        )


    line_of_sight = (
        differences
        / ranges[:, None]
    )


    H = np.column_stack(
        (
            -line_of_sight,
            np.ones(
                len(
                    satellite_positions
                )
            )
        )
    )


    return H


# ------------------------------------------------------------
# 9. POSITIONNEMENT GNSS PAR MOINDRES CARRES
# ------------------------------------------------------------

def solve_gnss_least_squares(
    satellite_positions,
    pseudoranges,
    initial_position,
    initial_clock_bias_meters=0.0,
    maximum_iterations=15,
    convergence_tolerance=1e-4
):
    """
    Estime :

        [x, y, z, b]

    par Gauss-Newton / moindres carres.
    """

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )


    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )


    if len(
        satellite_positions
    ) < 4:

        raise ValueError(
            "Au moins 4 satellites GNSS sont necessaires."
        )


    state = np.concatenate(
        (
            np.asarray(
                initial_position,
                dtype=float
            ),

            np.array([
                initial_clock_bias_meters
            ])
        )
    )


    converged = (
        False
    )


    correction_history = []


    for iteration in range(
        maximum_iterations
    ):

        receiver_position = (
            state[
                0:3
            ]
        )


        clock_bias_meters = (
            state[
                3
            ]
        )


        differences = (
            satellite_positions
            - receiver_position
        )


        geometric_ranges = np.linalg.norm(
            differences,
            axis=1
        )


        predicted_pseudoranges = (
            geometric_ranges
            + clock_bias_meters
        )


        residuals = (
            pseudoranges
            - predicted_pseudoranges
        )


        H = build_gnss_geometry_matrix(
            receiver_position=
                receiver_position,

            satellite_positions=
                satellite_positions
        )


        rank = np.linalg.matrix_rank(
            H
        )


        if rank < 4:

            raise RuntimeError(
                "Geometrie GNSS de rang insuffisant."
            )


        correction, _, _, _ = (
            np.linalg.lstsq(
                H,
                residuals,
                rcond=None
            )
        )


        state = (
            state
            + correction
        )


        correction_history.append(
            correction.copy()
        )


        if np.linalg.norm(
            correction
        ) < convergence_tolerance:

            converged = (
                True
            )

            break


    final_position = (
        state[
            0:3
        ]
    )


    final_clock_bias_meters = (
        state[
            3
        ]
    )


    final_geometric_ranges = np.linalg.norm(
        satellite_positions
        - final_position,
        axis=1
    )


    final_predicted_pseudoranges = (
        final_geometric_ranges
        + final_clock_bias_meters
    )


    final_residuals = (
        pseudoranges
        - final_predicted_pseudoranges
    )


    residual_rms = np.sqrt(
        np.mean(
            final_residuals**2
        )
    )


    final_H = build_gnss_geometry_matrix(
        receiver_position=
            final_position,

        satellite_positions=
            satellite_positions
    )


    normal_matrix = (
        final_H.T
        @ final_H
    )


    condition_number = (
        np.linalg.cond(
            normal_matrix
        )
    )


    return {
        "position":
            final_position,

        "clock_bias_meters":
            final_clock_bias_meters,

        "clock_bias_seconds":
            final_clock_bias_meters
            / C_LIGHT,

        "converged":
            converged,

        "iterations":
            iteration + 1,

        "residuals":
            final_residuals,

        "residual_rms":
            residual_rms,

        "geometry_matrix":
            final_H,

        "normal_matrix_condition":
            condition_number,

        "correction_history":
            correction_history
    }


# ------------------------------------------------------------
# 10. COVARIANCE DE LA SOLUTION GNSS
# ------------------------------------------------------------

def compute_gnss_solution_covariance(
    receiver_position,
    satellite_positions,
    pseudorange_noise_std
):
    """
    Pour des erreurs pseudorange independantes
    de variance sigma_rho^2 :

        Cov[x,y,z,b]
            = sigma_rho^2 (H^T H)^-1

    Retourne notamment le bloc position 3x3.
    """

    H = build_gnss_geometry_matrix(
        receiver_position=
            receiver_position,

        satellite_positions=
            satellite_positions
    )


    if np.linalg.matrix_rank(
        H
    ) < 4:

        raise RuntimeError(
            "Impossible de calculer la covariance : rang < 4."
        )


    normal_matrix = (
        H.T
        @ H
    )


    condition_number = (
        np.linalg.cond(
            normal_matrix
        )
    )


    if (
        not np.isfinite(
            condition_number
        )
        or condition_number
        > 1e12
    ):

        raise RuntimeError(
            "Geometrie GNSS numeriquement mal conditionnee."
        )


    geometry_covariance = np.linalg.inv(
        normal_matrix
    )


    solution_covariance = (
        pseudorange_noise_std**2
        * geometry_covariance
    )


    position_covariance = (
        solution_covariance[
            0:3,
            0:3
        ]
    )


    pdop = np.sqrt(
        np.trace(
            geometry_covariance[
                0:3,
                0:3
            ]
        )
    )


    tdop = np.sqrt(
        geometry_covariance[
            3,
            3
        ]
    )


    gdop = np.sqrt(
        pdop**2
        + tdop**2
    )


    return {
        "full_covariance":
            solution_covariance,

        "position_covariance":
            position_covariance,

        "pdop":
            pdop,

        "tdop":
            tdop,

        "gdop":
            gdop,

        "condition_number":
            condition_number
    }