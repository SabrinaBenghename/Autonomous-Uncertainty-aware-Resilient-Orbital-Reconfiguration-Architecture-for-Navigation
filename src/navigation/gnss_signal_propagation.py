import numpy as np

from src.dynamics.orbit import (
    MU_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.navigation.gnss_ekf_interface import (
    C_LIGHT,
    GPS_SEMI_MAJOR_AXIS,
    GPS_INCLINATION,
    GPS_NUMBER_OF_PLANES,
    GPS_SATELLITES_PER_PLANE
)


# ============================================================
# AURORA
# GNSS signal propagation / transmit-time model
#
# The signal received at t_rx was transmitted at:
#
#     t_tx = t_rx - tau
#
# with:
#
#     tau = range / c
#
# Therefore the GNSS satellite position used in the
# pseudorange model must be evaluated at t_tx.
# ============================================================


# ------------------------------------------------------------
# 1. SATELLITE ID -> ORBITAL INDICES
# ------------------------------------------------------------

def gps_id_to_indices(
    satellite_id
):
    """
    Convertit par exemple :

        GPS01 -> plane 0, satellite 0
        GPS04 -> plane 0, satellite 3
        GPS05 -> plane 1, satellite 0
    """

    satellite_number = int(
        satellite_id.replace(
            "GPS",
            ""
        )
    )


    zero_based_index = (
        satellite_number
        - 1
    )


    maximum_satellites = (
        GPS_NUMBER_OF_PLANES
        * GPS_SATELLITES_PER_PLANE
    )


    if (
        zero_based_index < 0
        or
        zero_based_index >= maximum_satellites
    ):

        raise ValueError(
            f"Identifiant GNSS invalide : {satellite_id}"
        )


    plane_index = (
        zero_based_index
        // GPS_SATELLITES_PER_PLANE
    )


    satellite_index = (
        zero_based_index
        % GPS_SATELLITES_PER_PLANE
    )


    return (
        plane_index,
        satellite_index
    )


# ------------------------------------------------------------
# 2. ETAT D'UN SATELLITE GPS SIMPLIFIE
# ------------------------------------------------------------

def get_simplified_gps_satellite_state(
    satellite_id,
    time_seconds
):
    """
    Retourne position et vitesse ECI du satellite GPS
    simplifie a une epoque arbitraire.
    """

    (
        plane_index,
        satellite_index
    ) = gps_id_to_indices(
        satellite_id
    )


    raan = (
        2.0
        * np.pi
        * plane_index
        / GPS_NUMBER_OF_PLANES
    )


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


    mean_motion = np.sqrt(
        MU_EARTH
        / GPS_SEMI_MAJOR_AXIS**3
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
        position,
        velocity
    ) = keplerian_to_cartesian(
        GPS_SEMI_MAJOR_AXIS,
        0.0,
        GPS_INCLINATION,
        raan,
        0.0,
        true_anomaly
    )


    return (
        position,
        velocity
    )


# ------------------------------------------------------------
# 3. GEOMETRIE AU TEMPS D'EMISSION
# ------------------------------------------------------------

def compute_transmit_time_geometry(
    receiver_position,
    satellite_ids,
    reception_time_seconds,
    number_of_light_time_iterations=5
):
    """
    Calcule pour chaque satellite :

        t_tx = t_rx - tau

    de maniere iterative.

    Le recepteur est considere a sa position
    au temps de reception.

    Le satellite est evalue a son temps
    d'emission.
    """

    receiver_position = np.asarray(
        receiver_position,
        dtype=float
    )


    transmit_positions = []

    reception_positions = []

    geometric_ranges = []

    travel_times = []

    satellite_displacements = []


    for satellite_id in satellite_ids:

        (
            satellite_position_rx,
            _
        ) = get_simplified_gps_satellite_state(
            satellite_id=
                satellite_id,

            time_seconds=
                reception_time_seconds
        )


        initial_range = np.linalg.norm(
            satellite_position_rx
            - receiver_position
        )


        travel_time = (
            initial_range
            / C_LIGHT
        )


        satellite_position_tx = (
            satellite_position_rx.copy()
        )


        for _ in range(
            number_of_light_time_iterations
        ):

            transmit_time = (
                reception_time_seconds
                - travel_time
            )


            (
                satellite_position_tx,
                _
            ) = get_simplified_gps_satellite_state(
                satellite_id=
                    satellite_id,

                time_seconds=
                    transmit_time
            )


            current_range = np.linalg.norm(
                satellite_position_tx
                - receiver_position
            )


            travel_time = (
                current_range
                / C_LIGHT
            )


        final_range = np.linalg.norm(
            satellite_position_tx
            - receiver_position
        )


        displacement = np.linalg.norm(
            satellite_position_rx
            - satellite_position_tx
        )


        transmit_positions.append(
            satellite_position_tx
        )


        reception_positions.append(
            satellite_position_rx
        )


        geometric_ranges.append(
            final_range
        )


        travel_times.append(
            travel_time
        )


        satellite_displacements.append(
            displacement
        )


    return {
        "transmit_positions":
            np.asarray(
                transmit_positions
            ),

        "reception_positions":
            np.asarray(
                reception_positions
            ),

        "geometric_ranges":
            np.asarray(
                geometric_ranges
            ),

        "travel_times":
            np.asarray(
                travel_times
            ),

        "satellite_displacements":
            np.asarray(
                satellite_displacements
            )
    }


# ------------------------------------------------------------
# 4. PSEUDORANGES AVEC TEMPS DE PROPAGATION
# ------------------------------------------------------------

def simulate_pseudoranges_with_transmit_time(
    receiver_position,
    satellite_ids,
    reception_time_seconds,
    receiver_clock_bias_seconds,
    pseudorange_noise_std,
    rng
):
    """
    Modele :

        rho_i =
            ||r_sat(t_tx) - r_receiver(t_rx)||
            + c * dt_receiver
            + noise_i
    """

    geometry = (
        compute_transmit_time_geometry(
            receiver_position=
                receiver_position,

            satellite_ids=
                satellite_ids,

            reception_time_seconds=
                reception_time_seconds
        )
    )


    clock_bias_meters = (
        C_LIGHT
        * receiver_clock_bias_seconds
    )


    noise = rng.normal(
        loc=0.0,
        scale=pseudorange_noise_std,
        size=len(
            satellite_ids
        )
    )


    pseudoranges = (
        geometry[
            "geometric_ranges"
        ]
        + clock_bias_meters
        + noise
    )


    return {
        "pseudoranges":
            pseudoranges,

        "clock_bias_meters":
            clock_bias_meters,

        "noise":
            noise,

        **geometry
    }


# ------------------------------------------------------------
# 5. MATRICE GEOMETRIQUE AU TEMPS D'EMISSION
# ------------------------------------------------------------

def build_transmit_time_geometry_matrix(
    receiver_position,
    satellite_ids,
    reception_time_seconds
):
    """
    Jacobienne approximative de la pseudorange
    par rapport a :

        [x, y, z, b]

    Les positions GNSS sont recalculees au
    temps d'emission.
    """

    geometry = (
        compute_transmit_time_geometry(
            receiver_position=
                receiver_position,

            satellite_ids=
                satellite_ids,

            reception_time_seconds=
                reception_time_seconds
        )
    )


    satellite_positions = (
        geometry[
            "transmit_positions"
        ]
    )


    differences = (
        satellite_positions
        - receiver_position
    )


    ranges = np.linalg.norm(
        differences,
        axis=1
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
                    satellite_ids
                )
            )
        )
    )


    return (
        H,
        geometry
    )


# ------------------------------------------------------------
# 6. SOLVEUR LS AVEC TEMPS DE PROPAGATION
# ------------------------------------------------------------

def solve_gnss_least_squares_transmit_time(
    satellite_ids,
    pseudoranges,
    reception_time_seconds,
    initial_position,
    initial_clock_bias_meters=0.0,
    maximum_iterations=15,
    convergence_tolerance=1e-4
):
    """
    Estimation [x, y, z, b] avec recalcul
    du temps d'emission a chaque iteration
    de Gauss-Newton.
    """

    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )


    if len(
        satellite_ids
    ) < 4:

        raise ValueError(
            "Au moins 4 satellites sont necessaires."
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


        (
            H,
            geometry
        ) = build_transmit_time_geometry_matrix(
            receiver_position=
                receiver_position,

            satellite_ids=
                satellite_ids,

            reception_time_seconds=
                reception_time_seconds
        )


        predicted_pseudoranges = (
            geometry[
                "geometric_ranges"
            ]
            + clock_bias_meters
        )


        residuals = (
            pseudoranges
            - predicted_pseudoranges
        )


        if np.linalg.matrix_rank(
            H
        ) < 4:

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


    (
        final_H,
        final_geometry
    ) = build_transmit_time_geometry_matrix(
        receiver_position=
            final_position,

        satellite_ids=
            satellite_ids,

        reception_time_seconds=
            reception_time_seconds
    )


    final_predicted_pseudoranges = (
        final_geometry[
            "geometric_ranges"
        ]
        + final_clock_bias_meters
    )


    final_residuals = (
        pseudoranges
        - final_predicted_pseudoranges
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
            np.sqrt(
                np.mean(
                    final_residuals**2
                )
            ),

        "geometry_matrix":
            final_H,

        "geometry":
            final_geometry
    }


# ------------------------------------------------------------
# 7. COVARIANCE GNSS
# ------------------------------------------------------------

def compute_transmit_time_solution_covariance(
    receiver_position,
    satellite_ids,
    reception_time_seconds,
    pseudorange_noise_std
):
    """
    Covariance lineaire :

        Cov =
            sigma_rho^2
            * (H^T H)^-1
    """

    (
        H,
        geometry
    ) = build_transmit_time_geometry_matrix(
        receiver_position=
            receiver_position,

        satellite_ids=
            satellite_ids,

        reception_time_seconds=
            reception_time_seconds
    )


    normal_matrix = (
        H.T
        @ H
    )


    condition_number = np.linalg.cond(
        normal_matrix
    )


    if (
        not np.isfinite(
            condition_number
        )
        or condition_number
        > 1e12
    ):

        raise RuntimeError(
            "Geometrie GNSS mal conditionnee."
        )


    geometry_covariance = np.linalg.inv(
        normal_matrix
    )


    covariance = (
        pseudorange_noise_std**2
        * geometry_covariance
    )


    return {
        "full_covariance":
            covariance,

        "position_covariance":
            covariance[
                0:3,
                0:3
            ],

        "pdop":
            np.sqrt(
                np.trace(
                    geometry_covariance[
                        0:3,
                        0:3
                    ]
                )
            ),

        "geometry":
            geometry,

        "condition_number":
            condition_number
    }