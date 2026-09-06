import numpy as np
import pandas as pd

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    propagate_orbit_with_j2_and_drag_like
)

from src.navigation.gnss_ekf_interface import (
    generate_simplified_gps_positions,
    select_visible_gnss_satellites
)

from src.navigation.gnss_signal_propagation import (
    compute_transmit_time_solution_covariance
)


# ============================================================
# AURORA
# Phase 17-A
#
# Orbital scenario / GNSS geometry sweep
#
#
# Purpose:
#
# Study how GNSS geometry changes when varying:
#
#       altitude
#       inclination
#       initial orbital phase
#
#
# Outputs:
#
#       visible satellites
#       >= 4 satellite availability
#       >= 6 satellite redundancy availability
#       PDOP statistics
#       longest low-redundancy intervals
#
#
# IMPORTANT:
#
# >= 6 satellites is an AURORA-specific redundancy
# requirement for the current leave-one-out FDIR architecture.
#
# It is NOT a universal GNSS requirement.
#
#
# This phase does NOT yet run:
#
#       EKF
#       attitude MEKF
#       faults
#       AI
#
# Those are introduced after the geometry domain is mapped.
# ============================================================


# ------------------------------------------------------------
# 1. CONSTANTS
# ------------------------------------------------------------

MU_EARTH = (
    3.986004418e14
)


REFERENCE_ECCENTRICITY = (
    0.01
)


REFERENCE_RAAN_DEG = (
    40.0
)


REFERENCE_ARGUMENT_OF_PERIAPSIS_DEG = (
    30.0
)


DEFAULT_PSEUDORANGE_NOISE_STD_M = (
    3.0
)


DEFAULT_NUMBER_OF_ORBITS = (
    2.0
)


DEFAULT_GEOMETRY_STEP_SECONDS = (
    30.0
)


# ------------------------------------------------------------
# 2. PHASE-17 DESIGN DOMAIN
# ------------------------------------------------------------

ALTITUDES_KM = [
    400.0,
    550.0,
    700.0
]


INCLINATIONS_DEG = [
    0.0,
    30.0,
    60.0,
    75.0,
    90.0,
    97.6
]


INITIAL_PHASES_DEG = [
    0.0,
    60.0,
    120.0,
    180.0,
    240.0,
    300.0
]


# ------------------------------------------------------------
# 3. SCENARIO GENERATOR
# ------------------------------------------------------------

def generate_phase17_geometry_scenarios():

    rows = []

    scenario_index = (
        0
    )


    for altitude_km in ALTITUDES_KM:

        for inclination_deg in INCLINATIONS_DEG:

            for initial_phase_deg in INITIAL_PHASES_DEG:

                rows.append(
                    {
                        "scenario_id":
                            (
                                f"AURORA17_"
                                f"{scenario_index:03d}"
                            ),

                        "scenario_index":
                            scenario_index,

                        "altitude_km":
                            float(
                                altitude_km
                            ),

                        "inclination_deg":
                            float(
                                inclination_deg
                            ),

                        "initial_true_anomaly_deg":
                            float(
                                initial_phase_deg
                            ),

                        "eccentricity":
                            REFERENCE_ECCENTRICITY,

                        "raan_deg":
                            REFERENCE_RAAN_DEG,

                        "argument_of_periapsis_deg":
                            REFERENCE_ARGUMENT_OF_PERIAPSIS_DEG
                    }
                )


                scenario_index += (
                    1
                )


    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# 4. ORBIT PERIOD
# ------------------------------------------------------------

def compute_two_body_period_seconds(
    semi_major_axis_m
):

    return float(
        2.0
        *
        np.pi
        *
        np.sqrt(
            semi_major_axis_m**3
            /
            MU_EARTH
        )
    )


# ------------------------------------------------------------
# 5. LONGEST FALSE INTERVAL
#
# Example:
#
# condition = satellites >= 6
#
# We return the longest interval during which condition=False.
# ------------------------------------------------------------

def longest_unavailable_interval_seconds(
    condition,
    time_seconds
):

    condition = np.asarray(
        condition,
        dtype=bool
    )


    time_seconds = np.asarray(
        time_seconds,
        dtype=float
    )


    if len(
        condition
    ) != len(
        time_seconds
    ):

        raise ValueError(
            "Condition/time length mismatch."
        )


    if len(
        condition
    ) < 2:

        return (
            0.0
        )


    maximum_duration = (
        0.0
    )


    current_start_index = (
        None
    )


    for index, available in enumerate(
        condition
    ):

        if not available:

            if current_start_index is None:

                current_start_index = (
                    index
                )


        else:

            if current_start_index is not None:

                end_index = (
                    index
                    -
                    1
                )


                if end_index > current_start_index:

                    duration = (
                        time_seconds[
                            end_index
                        ]
                        -
                        time_seconds[
                            current_start_index
                        ]
                    )

                else:

                    duration = (
                        0.0
                    )


                maximum_duration = max(
                    maximum_duration,
                    duration
                )


                current_start_index = (
                    None
                )


    # ========================================================
    # Sequence ending unavailable
    # ========================================================

    if current_start_index is not None:

        end_index = (
            len(
                condition
            )
            -
            1
        )


        if end_index > current_start_index:

            duration = (
                time_seconds[
                    end_index
                ]
                -
                time_seconds[
                    current_start_index
                ]
            )

        else:

            duration = (
                0.0
            )


        maximum_duration = max(
            maximum_duration,
            duration
        )


    return float(
        maximum_duration
    )


# ------------------------------------------------------------
# 6. SAFE PERCENTILE
# ------------------------------------------------------------

def _safe_percentile(
    values,
    percentile
):

    values = np.asarray(
        values,
        dtype=float
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return (
            np.nan
        )


    return float(
        np.percentile(
            values,
            percentile
        )
    )


# ------------------------------------------------------------
# 7. RUN ONE GEOMETRY SCENARIO
# ------------------------------------------------------------

def run_orbit_geometry_scenario(
    scenario,
    number_of_orbits=
        DEFAULT_NUMBER_OF_ORBITS,
    geometry_step_seconds=
        DEFAULT_GEOMETRY_STEP_SECONDS,
    pseudorange_noise_std_m=
        DEFAULT_PSEUDORANGE_NOISE_STD_M
):

    scenario = dict(
        scenario
    )


    altitude_m = (
        float(
            scenario[
                "altitude_km"
            ]
        )
        *
        1000.0
    )


    semi_major_axis_m = (
        R_EARTH
        +
        altitude_m
    )


    eccentricity = float(
        scenario[
            "eccentricity"
        ]
    )


    inclination_rad = np.deg2rad(
        float(
            scenario[
                "inclination_deg"
            ]
        )
    )


    raan_rad = np.deg2rad(
        float(
            scenario[
                "raan_deg"
            ]
        )
    )


    argument_of_periapsis_rad = np.deg2rad(
        float(
            scenario[
                "argument_of_periapsis_deg"
            ]
        )
    )


    true_anomaly_rad = np.deg2rad(
        float(
            scenario[
                "initial_true_anomaly_deg"
            ]
        )
    )


    # ========================================================
    # Initial state
    # ========================================================

    (
        initial_position,
        initial_velocity
    ) = keplerian_to_cartesian(
        semi_major_axis_m,
        eccentricity,
        inclination_rad,
        raan_rad,
        argument_of_periapsis_rad,
        true_anomaly_rad
    )


    initial_state = np.concatenate(
        (
            initial_position,
            initial_velocity
        )
    )


    # ========================================================
    # Duration
    # ========================================================

    orbital_period_seconds = (
        compute_two_body_period_seconds(
            semi_major_axis_m
        )
    )


    simulation_duration_seconds = (
        float(
            number_of_orbits
        )
        *
        orbital_period_seconds
    )


    number_of_points = (
        int(
            np.round(
                simulation_duration_seconds
                /
                float(
                    geometry_step_seconds
                )
            )
        )
        +
        1
    )


    number_of_points = max(
        number_of_points,
        2
    )


    # ========================================================
    # J2 truth propagation
    #
    # base_acceleration = 0 removes the synthetic
    # non-gravitational force from this geometry-only sweep.
    # ========================================================

    solution = (
        propagate_orbit_with_j2_and_drag_like(
            initial_state=
                initial_state,

            duration=
                simulation_duration_seconds,

            number_of_points=
                number_of_points,

            base_acceleration=
                0.0
        )
    )


    time_seconds = np.asarray(
        solution.t,
        dtype=float
    )


    states = (
        solution.y.T
    )


    if len(
        time_seconds
    ) != number_of_points:

        raise RuntimeError(
            "Unexpected propagator output length."
        )


    if len(
        states
    ) != number_of_points:

        raise RuntimeError(
            "Unexpected state-history length."
        )


    # ========================================================
    # Geometry histories
    # ========================================================

    visible_satellite_count = np.zeros(
        number_of_points,
        dtype=int
    )


    pdop_history = np.full(
        number_of_points,
        np.nan
    )


    geometry_failures = (
        0
    )


    for epoch_index in range(
        number_of_points
    ):

        reception_time_seconds = float(
            time_seconds[
                epoch_index
            ]
        )


        receiver_position = (
            states[
                epoch_index,
                0:3
            ]
        )


        (
            all_satellite_positions,
            all_satellite_ids
        ) = generate_simplified_gps_positions(
            time_seconds=
                reception_time_seconds
        )


        (
            _,
            visible_ids
        ) = select_visible_gnss_satellites(
            receiver_position=
                receiver_position,

            satellite_positions=
                all_satellite_positions,

            satellite_ids=
                all_satellite_ids
        )


        visible_ids = [
            str(
                satellite_id
            ).strip()

            for satellite_id in visible_ids
        ]


        visible_satellite_count[
            epoch_index
        ] = (
            len(
                visible_ids
            )
        )


        if len(
            visible_ids
        ) < 4:

            continue


        try:

            covariance_result = (
                compute_transmit_time_solution_covariance(
                    receiver_position=
                        receiver_position,

                    satellite_ids=
                        visible_ids,

                    reception_time_seconds=
                        reception_time_seconds,

                    pseudorange_noise_std=
                        float(
                            pseudorange_noise_std_m
                        )
                )
            )


            pdop_history[
                epoch_index
            ] = float(
                covariance_result[
                    "pdop"
                ]
            )


        except (
            RuntimeError,
            ValueError,
            KeyError,
            np.linalg.LinAlgError
        ):

            geometry_failures += (
                1
            )


    # ========================================================
    # Availability
    # ========================================================

    four_sat_available = (
        visible_satellite_count
        >=
        4
    )


    six_sat_available = (
        visible_satellite_count
        >=
        6
    )


    pdop_valid = np.isfinite(
        pdop_history
    )


    # ========================================================
    # Metrics
    # ========================================================

    mean_visible_satellites = float(
        np.mean(
            visible_satellite_count
        )
    )


    minimum_visible_satellites = int(
        np.min(
            visible_satellite_count
        )
    )


    maximum_visible_satellites = int(
        np.max(
            visible_satellite_count
        )
    )


    four_satellite_availability = float(
        100.0
        *
        np.mean(
            four_sat_available
        )
    )


    six_satellite_availability = float(
        100.0
        *
        np.mean(
            six_sat_available
        )
    )


    pdop_solution_availability = float(
        100.0
        *
        np.mean(
            pdop_valid
        )
    )


    valid_pdop = (
        pdop_history[
            pdop_valid
        ]
    )


    if len(
        valid_pdop
    ) > 0:

        pdop_mean = float(
            np.mean(
                valid_pdop
            )
        )


        pdop_min = float(
            np.min(
                valid_pdop
            )
        )


        pdop_max = float(
            np.max(
                valid_pdop
            )
        )


    else:

        pdop_mean = np.nan
        pdop_min = np.nan
        pdop_max = np.nan


    longest_below_four_seconds = (
        longest_unavailable_interval_seconds(
            condition=
                four_sat_available,

            time_seconds=
                time_seconds
        )
    )


    longest_below_six_seconds = (
        longest_unavailable_interval_seconds(
            condition=
                six_sat_available,

            time_seconds=
                time_seconds
        )
    )


    actual_mean_step_seconds = float(
        np.mean(
            np.diff(
                time_seconds
            )
        )
    )


    return {
        "scenario_id":
            str(
                scenario[
                    "scenario_id"
                ]
            ),

        "scenario_index":
            int(
                scenario[
                    "scenario_index"
                ]
            ),

        "altitude_km":
            float(
                scenario[
                    "altitude_km"
                ]
            ),

        "inclination_deg":
            float(
                scenario[
                    "inclination_deg"
                ]
            ),

        "initial_true_anomaly_deg":
            float(
                scenario[
                    "initial_true_anomaly_deg"
                ]
            ),

        "semi_major_axis_km":
            float(
                semi_major_axis_m
                /
                1000.0
            ),

        "orbital_period_min":
            float(
                orbital_period_seconds
                /
                60.0
            ),

        "number_of_orbits":
            float(
                number_of_orbits
            ),

        "epochs":
            int(
                number_of_points
            ),

        "mean_geometry_step_s":
            actual_mean_step_seconds,

        "visible_satellites_min":
            minimum_visible_satellites,

        "visible_satellites_mean":
            mean_visible_satellites,

        "visible_satellites_max":
            maximum_visible_satellites,

        "availability_ge4_percent":
            four_satellite_availability,

        "availability_ge6_percent":
            six_satellite_availability,

        "pdop_availability_percent":
            pdop_solution_availability,

        "pdop_min":
            pdop_min,

        "pdop_mean":
            pdop_mean,

        "pdop_p90":
            _safe_percentile(
                valid_pdop,
                90.0
            ),

        "pdop_p95":
            _safe_percentile(
                valid_pdop,
                95.0
            ),

        "pdop_max":
            pdop_max,

        "longest_below4_s":
            longest_below_four_seconds,

        "longest_below6_s":
            longest_below_six_seconds,

        "longest_below4_min":
            float(
                longest_below_four_seconds
                /
                60.0
            ),

        "longest_below6_min":
            float(
                longest_below_six_seconds
                /
                60.0
            ),

        "geometry_failures":
            int(
                geometry_failures
            )
    }