from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    t as student_t
)

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
    simulate_pseudoranges_with_transmit_time,
    compute_transmit_time_solution_covariance
)

from src.navigation.gnss_fdir_transmit_time import (
    solve_gnss_with_fdir_transmit_time
)

from src.ai.gnss_anomaly_features import (
    extract_gnss_anomaly_features
)

from src.ai.gnss_temporal_features import (
    TEMPORAL_FEATURE_COLUMNS,
    add_causal_temporal_features,
    temporal_feature_valid_mask
)

from src.ai.isolation_forest_detector import (
    IsolationForestAnomalyDetector
)

from src.ai.hybrid_anomaly_supervisor import (
    HybridGnssAnomalySupervisor
)


# ============================================================
# AURORA
# Experience 015-E
#
# FRESH HELD-OUT MONTE CARLO VALIDATION
#
#
# Goal:
#
# Statistically validate the final hybrid:
#
#       Classical FDIR
#           +
#       Temporal Isolation Forest
#           +
#       2-out-of-3 persistence supervisor
#
#
# IMPORTANT:
#
# The AI model is trained/calibrated using the OLD
# 015-C nominal dataset:
#
#       training:
#           runs 0, 1, 2
#
#       calibration:
#           run 3
#
#
# Evaluation is performed on 20 completely NEW runs:
#
#       run IDs 100 ... 119
#
# with NEW:
#
#       pseudorange-noise realizations
#       navigation-prior errors
#
#
# Therefore:
#
#       training/calibration/test are disjoint.
#
#
# Scenario:
#
#       100-120 min sequence
#       exactly 5 satellites
#
#       persistent fault:
#           105-115 min
#
#       amplitudes:
#           5, 10, 20, 50 m
#
#
# This is NOT yet the broad Phase-16 robustness campaign.
#
# Orbit geometry and fault timeline remain fixed here.
# 015-E specifically validates generalization across
# fresh measurement-noise/prior realizations.
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

training_dataset_path = (
    Path("data")
    /
    "phase15"
    /
    "gnss_temporal_dataset_015c.csv"
)


if not training_dataset_path.exists():

    raise FileNotFoundError(
        "Dataset 015-C introuvable : "
        f"{training_dataset_path}"
    )


data_directory = (
    Path("data")
    /
    "phase15"
)


table_directory = (
    Path("results")
    /
    "tables"
)


figure_directory = (
    Path("results")
    /
    "figures"
)


data_directory.mkdir(
    parents=True,
    exist_ok=True
)


table_directory.mkdir(
    parents=True,
    exist_ok=True
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


heldout_dataset_path = (
    data_directory
    /
    "gnss_hybrid_heldout_dataset_015e.csv"
)


run_metrics_path = (
    table_directory
    /
    "phase15_015e_run_metrics.csv"
)


amplitude_summary_path = (
    table_directory
    /
    "phase15_015e_amplitude_summary.csv"
)


# ------------------------------------------------------------
# 2. MONTE CARLO
# ------------------------------------------------------------

number_of_test_runs = (
    20
)


heldout_run_ids = list(
    range(
        100,
        100
        +
        number_of_test_runs
    )
)


training_run_ids = [
    0,
    1,
    2
]


calibration_run_ids = [
    3
]


# ------------------------------------------------------------
# 3. TIME
# ------------------------------------------------------------

simulation_duration_minutes = (
    120.0
)


simulation_duration_seconds = (
    simulation_duration_minutes
    *
    60.0
)


dt = (
    10.0
)


number_of_epochs = (
    int(
        simulation_duration_seconds
        /
        dt
    )
    +
    1
)


time = np.linspace(
    0.0,
    simulation_duration_seconds,
    number_of_epochs
)


time_minutes = (
    time
    /
    60.0
)


sequence_start_minutes = (
    100.0
)


sequence_end_minutes = (
    120.0
)


sequence_mask = (
    (
        time_minutes
        >=
        sequence_start_minutes
    )
    &
    (
        time_minutes
        <=
        sequence_end_minutes
    )
)


sequence_indices = np.where(
    sequence_mask
)[0]


fault_start_minutes = (
    105.0
)


fault_end_minutes = (
    115.0
)


fault_start_seconds = (
    fault_start_minutes
    *
    60.0
)


# ------------------------------------------------------------
# 4. GNSS PARAMETERS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


speed_of_light = (
    299_792_458.0
)


true_clock_bias_meters = (
    speed_of_light
    *
    receiver_clock_bias_seconds
)


gnss_confidence = (
    0.99
)


preferred_fault_satellite = (
    "GPS04"
)


fault_magnitudes_m = [
    0.0,
    5.0,
    10.0,
    20.0,
    50.0
]


# ------------------------------------------------------------
# 5. PRIOR MODEL
# ------------------------------------------------------------

prior_position_sigma_m = (
    20.0
)


prior_clock_sigma_m = (
    10.0
)


# ------------------------------------------------------------
# 6. ORBIT
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH
    +
    550_000.0
)


eccentricity = (
    0.01
)


inclination = np.deg2rad(
    97.6
)


raan = np.deg2rad(
    40.0
)


argument_of_periapsis = np.deg2rad(
    30.0
)


true_anomaly = np.deg2rad(
    25.0
)


(
    initial_position,
    initial_velocity
) = keplerian_to_cartesian(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly
)


initial_state = np.concatenate(
    (
        initial_position,
        initial_velocity
    )
)


truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            initial_state,

        duration=
            simulation_duration_seconds,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            2.0e-5
    )
)


truth_states = (
    truth_solution.y.T
)


maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        -
        time
    )
)


# ------------------------------------------------------------
# 7. HELPER — FIVE SATELLITES
# ------------------------------------------------------------

def select_five_satellites(
    visible_ids
):

    visible_ids = [
        str(
            satellite_id
        ).strip()
        for satellite_id in visible_ids
    ]


    if len(
        visible_ids
    ) < 5:

        return None


    if (
        preferred_fault_satellite
        in
        visible_ids
    ):

        other_ids = [
            satellite_id
            for satellite_id in visible_ids
            if satellite_id
            !=
            preferred_fault_satellite
        ]


        return (
            [
                preferred_fault_satellite
            ]
            +
            other_ids[
                :4
            ]
        )


    return (
        visible_ids[
            :5
        ]
    )


# ------------------------------------------------------------
# 8. HELPER — FAULT TARGET
# ------------------------------------------------------------

def choose_fault_target(
    satellite_ids
):

    if (
        preferred_fault_satellite
        in
        satellite_ids
    ):

        return (
            preferred_fault_satellite
        )


    return (
        satellite_ids[
            0
        ]
    )


# ------------------------------------------------------------
# 9. LOAD TRAINING / CALIBRATION DATA
# ------------------------------------------------------------

old_dataset = pd.read_csv(
    training_dataset_path
)


old_dataset = old_dataset[
    old_dataset[
        "temporal_features_valid"
    ]
    ==
    1
].copy()


training_nominal = old_dataset[
    (
        old_dataset[
            "run_id"
        ].isin(
            training_run_ids
        )
    )
    &
    (
        old_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


calibration_nominal = old_dataset[
    (
        old_dataset[
            "run_id"
        ].isin(
            calibration_run_ids
        )
    )
    &
    (
        old_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


# ------------------------------------------------------------
# 10. TRAIN FINAL TEMPORAL DETECTOR
# ------------------------------------------------------------

detector = (
    IsolationForestAnomalyDetector(
        feature_columns=
            TEMPORAL_FEATURE_COLUMNS,

        n_estimators=
            500,

        random_state=
            1511
    )
)


detector.fit_nominal(
    training_nominal
)


calibration = (
    detector.calibrate_threshold(
        nominal_calibration_dataframe=
            calibration_nominal,

        target_false_alarm_rate=
            0.01
    )
)


# ------------------------------------------------------------
# 11. HELD-OUT SPLIT VALIDATION
# ------------------------------------------------------------

training_and_calibration_ids = set(
    training_run_ids
    +
    calibration_run_ids
)


heldout_ids = set(
    heldout_run_ids
)


heldout_runs_disjoint = (
    len(
        training_and_calibration_ids.intersection(
            heldout_ids
        )
    )
    ==
    0
)


# ------------------------------------------------------------
# 12. GENERATE COMPLETELY NEW TEST SEQUENCES
# ------------------------------------------------------------

rows = []


print(
    "\n"
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-E"
)

print(
    "Fresh held-out Monte Carlo validation of hybrid GNSS anomaly detector"
)

print(
    "========================================================================================================================"
)

print(
    f"Held-out runs : "
    f"{number_of_test_runs}"
)

print(
    f"Run IDs : "
    f"{heldout_run_ids[0]}"
    f" ... "
    f"{heldout_run_ids[-1]}"
)

print(
    f"Temporal AI threshold : "
    f"{calibration['threshold']:.6f}"
)

print(
    f"Calibration P_FA : "
    f"{100.0 * calibration['achieved_false_alarm_rate']:.2f} %"
)

print(
    f"Held-out runs disjoint : "
    f"{heldout_runs_disjoint}"
)

print()

print(
    "Generation fresh test sequences:"
)


for monte_carlo_index, run_id in enumerate(
    heldout_run_ids
):

    # ========================================================
    # Completely new random seeds
    # ========================================================

    pseudorange_rng = np.random.default_rng(
        150_000
        +
        run_id
    )


    prior_rng = np.random.default_rng(
        160_000
        +
        run_id
    )


    # ========================================================
    # Fixed navigation-prior error for this run
    # ========================================================

    prior_position_error = (
        prior_rng.normal(
            0.0,
            prior_position_sigma_m,
            size=3
        )
    )


    prior_clock_error_m = float(
        prior_rng.normal(
            0.0,
            prior_clock_sigma_m
        )
    )


    rows_before = len(
        rows
    )


    for epoch_index in sequence_indices:

        reception_time_seconds = (
            time[
                epoch_index
            ]
        )


        reception_time_minutes = (
            time_minutes[
                epoch_index
            ]
        )


        receiver_position_true = (
            truth_states[
                epoch_index,
                0:3
            ]
        )


        prior_position = (
            receiver_position_true
            +
            prior_position_error
        )


        prior_clock_bias_meters = (
            true_clock_bias_meters
            +
            prior_clock_error_m
        )


        # ====================================================
        # Visible constellation
        # ====================================================

        (
            all_positions,
            all_ids
        ) = generate_simplified_gps_positions(
            time_seconds=
                reception_time_seconds
        )


        (
            _,
            visible_ids
        ) = select_visible_gnss_satellites(
            receiver_position=
                receiver_position_true,

            satellite_positions=
                all_positions,

            satellite_ids=
                all_ids
        )


        working_ids = (
            select_five_satellites(
                visible_ids
            )
        )


        if working_ids is None:

            continue


        # ====================================================
        # Same nominal pseudorange noise is reused across
        # all fault amplitudes at this epoch.
        # ====================================================

        nominal_result = (
            simulate_pseudoranges_with_transmit_time(
                receiver_position=
                    receiver_position_true,

                satellite_ids=
                    working_ids,

                reception_time_seconds=
                    reception_time_seconds,

                receiver_clock_bias_seconds=
                    receiver_clock_bias_seconds,

                pseudorange_noise_std=
                    pseudorange_noise_std,

                rng=
                    pseudorange_rng
            )
        )


        nominal_pseudoranges = np.asarray(
            nominal_result[
                "pseudoranges"
            ],
            dtype=float
        )


        fault_target = (
            choose_fault_target(
                working_ids
            )
        )


        fault_target_index = (
            working_ids.index(
                fault_target
            )
        )


        fault_window_active = (
            reception_time_minutes
            >=
            fault_start_minutes
            and
            reception_time_minutes
            <
            fault_end_minutes
        )


        # ====================================================
        # Paired amplitudes
        # ====================================================

        for fault_magnitude_m in fault_magnitudes_m:

            pseudoranges = (
                nominal_pseudoranges.copy()
            )


            actual_fault_active = (
                fault_window_active
                and
                fault_magnitude_m
                >
                0.0
            )


            if actual_fault_active:

                pseudoranges[
                    fault_target_index
                ] += (
                    fault_magnitude_m
                )


            # =================================================
            # Classical FDIR
            # =================================================

            try:

                fdir_result = (
                    solve_gnss_with_fdir_transmit_time(
                        satellite_ids=
                            working_ids,

                        pseudoranges=
                            pseudoranges,

                        reception_time_seconds=
                            reception_time_seconds,

                        initial_position=
                            prior_position,

                        initial_clock_bias_meters=
                            prior_clock_bias_meters,

                        pseudorange_noise_std=
                            pseudorange_noise_std,

                        confidence=
                            gnss_confidence
                    )
                )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError
            ):

                continue


            # =================================================
            # Full GNSS solution
            # =================================================

            full_solution = (
                fdir_result.get(
                    "full_solution",
                    None
                )
            )


            if full_solution is None:

                full_solution = (
                    fdir_result.get(
                        "solution",
                        None
                    )
                )


            if (
                full_solution is None
                or
                not full_solution.get(
                    "converged",
                    False
                )
            ):

                continue


            # =================================================
            # PDOP
            # =================================================

            try:

                covariance_result = (
                    compute_transmit_time_solution_covariance(
                        receiver_position=
                            full_solution[
                                "position"
                            ],

                        satellite_ids=
                            working_ids,

                        reception_time_seconds=
                            reception_time_seconds,

                        pseudorange_noise_std=
                            pseudorange_noise_std
                    )
                )


                pdop = float(
                    covariance_result[
                        "pdop"
                    ]
                )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
                KeyError
            ):

                continue


            # =================================================
            # Instantaneous AI features
            # =================================================

            features = (
                extract_gnss_anomaly_features(
                    fdir_result=
                        fdir_result,

                    number_of_satellites=
                        len(
                            working_ids
                        ),

                    pseudorange_noise_std=
                        pseudorange_noise_std,

                    confidence=
                        gnss_confidence,

                    pdop=
                        pdop,

                    prior_position=
                        prior_position,

                    prior_clock_bias_meters=
                        prior_clock_bias_meters
                )
            )


            if features is None:

                continue


            rows.append(
                {
                    "run_id":
                        run_id,

                    "epoch_index":
                        epoch_index,

                    "time_seconds":
                        reception_time_seconds,

                    "time_minutes":
                        reception_time_minutes,

                    "fault_magnitude_m":
                        float(
                            fault_magnitude_m
                        ),

                    "fault_active":
                        int(
                            actual_fault_active
                        ),

                    "fdir_detected":
                        int(
                            bool(
                                fdir_result.get(
                                    "fault_detected",
                                    False
                                )
                            )
                        ),

                    **features
                }
            )


    print(
        f"Run "
        f"{monte_carlo_index + 1:02d}/"
        f"{number_of_test_runs} : "
        f"{len(rows) - rows_before} rows"
    )


# ------------------------------------------------------------
# 13. TEMPORAL FEATURE ENGINEERING
# ------------------------------------------------------------

heldout_dataset = pd.DataFrame(
    rows
)


if len(
    heldout_dataset
) == 0:

    raise RuntimeError(
        "Le dataset held-out 015-E est vide."
    )


heldout_dataset = (
    add_causal_temporal_features(
        dataframe=
            heldout_dataset,

        group_columns=[
            "run_id",
            "fault_magnitude_m"
        ],

        time_column=
            "time_seconds"
    )
)


valid_temporal_mask = (
    temporal_feature_valid_mask(
        heldout_dataset
    )
)


heldout_dataset[
    "temporal_features_valid"
] = (
    valid_temporal_mask.astype(
        int
    )
)


valid_dataset = heldout_dataset[
    valid_temporal_mask
].copy()


# ------------------------------------------------------------
# 14. TEMPORAL FEATURE VALIDATION
# ------------------------------------------------------------

temporal_feature_matrix = (
    valid_dataset[
        TEMPORAL_FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )
)


all_temporal_features_finite = bool(
    np.all(
        np.isfinite(
            temporal_feature_matrix
        )
    )
)


# ------------------------------------------------------------
# 15. SCORE COMPLETELY HELD-OUT DATA
# ------------------------------------------------------------

valid_dataset = (
    valid_dataset.sort_values(
        by=[
            "run_id",
            "fault_magnitude_m",
            "time_seconds"
        ]
    )
    .reset_index(
        drop=True
    )
)


valid_dataset[
    "temporal_score"
] = (
    detector.score_samples(
        valid_dataset
    )
)


valid_dataset[
    "ai_raw_alarm"
] = (
    valid_dataset[
        "temporal_score"
    ]
    >=
    detector.threshold
).astype(
    int
)


# ------------------------------------------------------------
# 16. HYBRID SUPERVISOR
# ------------------------------------------------------------

ai_vote_window = (
    3
)


minimum_ai_votes = (
    2
)


number_of_valid_rows = len(
    valid_dataset
)


naive_or_alarm = np.zeros(
    number_of_valid_rows,
    dtype=int
)


hybrid_alarm = np.zeros(
    number_of_valid_rows,
    dtype=int
)


ai_confirmed = np.zeros(
    number_of_valid_rows,
    dtype=int
)


ai_vote_count = np.zeros(
    number_of_valid_rows,
    dtype=int
)


grouped_indices = (
    valid_dataset.groupby(
        [
            "run_id",
            "fault_magnitude_m"
        ],
        sort=False
    ).groups
)


for _, indices in grouped_indices.items():

    ordered_indices = sorted(
        list(
            indices
        ),
        key=
            lambda index:
                valid_dataset.loc[
                    index,
                    "time_seconds"
                ]
    )


    supervisor = (
        HybridGnssAnomalySupervisor(
            ai_vote_window=
                ai_vote_window,

            minimum_ai_votes=
                minimum_ai_votes
        )
    )


    for index in ordered_indices:

        classical_alarm = bool(
            valid_dataset.loc[
                index,
                "fdir_detected"
            ]
        )


        raw_ai_alarm = bool(
            valid_dataset.loc[
                index,
                "ai_raw_alarm"
            ]
        )


        decision = supervisor.update(
            classical_alarm=
                classical_alarm,

            ai_raw_alarm=
                raw_ai_alarm
        )


        naive_or_alarm[
            index
        ] = int(
            classical_alarm
            or
            raw_ai_alarm
        )


        hybrid_alarm[
            index
        ] = int(
            decision.hard_alarm
        )


        ai_confirmed[
            index
        ] = int(
            decision.ai_confirmed
        )


        ai_vote_count[
            index
        ] = int(
            decision.ai_vote_count
        )


valid_dataset[
    "naive_or_alarm"
] = (
    naive_or_alarm
)


valid_dataset[
    "hybrid_alarm"
] = (
    hybrid_alarm
)


valid_dataset[
    "ai_confirmed"
] = (
    ai_confirmed
)


valid_dataset[
    "ai_vote_count"
] = (
    ai_vote_count
)


# ------------------------------------------------------------
# 17. HELPER — ALARM EPISODES
# ------------------------------------------------------------

def count_alarm_episodes(
    dataframe,
    prediction_column
):

    alarms = (
        dataframe.sort_values(
            "time_seconds"
        )[
            prediction_column
        ]
        .to_numpy(
            dtype=int
        )
    )


    episodes = (
        0
    )


    previous_alarm = (
        0
    )


    for alarm in alarms:

        if (
            alarm
            ==
            1
            and
            previous_alarm
            ==
            0
        ):

            episodes += (
                1
            )


        previous_alarm = (
            alarm
        )


    return (
        episodes
    )


# ------------------------------------------------------------
# 18. HELPER — FIRST ALARM LATENCY
# ------------------------------------------------------------

def compute_first_alarm_latency(
    dataframe,
    prediction_column
):

    dataframe = dataframe.sort_values(
        "time_seconds"
    )


    predictions = (
        dataframe[
            prediction_column
        ].to_numpy(
            dtype=bool
        )
    )


    alarm_indices = np.where(
        predictions
    )[0]


    if len(
        alarm_indices
    ) == 0:

        return np.nan


    times = (
        dataframe[
            "time_seconds"
        ].to_numpy(
            dtype=float
        )
    )


    return (
        times[
            alarm_indices[
                0
            ]
        ]
        -
        fault_start_seconds
    )


# ------------------------------------------------------------
# 19. PER-RUN METRICS
#
# We calculate each run independently before doing
# ensemble statistics.
# ------------------------------------------------------------

run_metric_rows = []


for run_id in heldout_run_ids:

    run_dataset = valid_dataset[
        valid_dataset[
            "run_id"
        ]
        ==
        run_id
    ]


    nominal_subset = run_dataset[
        run_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    ]


    all_fault_subset = run_dataset[
        run_dataset[
            "fault_active"
        ]
        ==
        1
    ]


    row = {
        "run_id":
            run_id,

        # ====================================================
        # Nominal false alarm rate
        # ====================================================

        "fdir_pfa":
            100.0
            *
            np.mean(
                nominal_subset[
                    "fdir_detected"
                ]
            ),

        "temporal_ai_pfa":
            100.0
            *
            np.mean(
                nominal_subset[
                    "ai_raw_alarm"
                ]
            ),

        "naive_or_pfa":
            100.0
            *
            np.mean(
                nominal_subset[
                    "naive_or_alarm"
                ]
            ),

        "hybrid_pfa":
            100.0
            *
            np.mean(
                nominal_subset[
                    "hybrid_alarm"
                ]
            ),

        # ====================================================
        # False alarm episodes
        # ====================================================

        "fdir_false_alarm_episodes":
            count_alarm_episodes(
                nominal_subset,
                "fdir_detected"
            ),

        "temporal_ai_false_alarm_episodes":
            count_alarm_episodes(
                nominal_subset,
                "ai_raw_alarm"
            ),

        "naive_or_false_alarm_episodes":
            count_alarm_episodes(
                nominal_subset,
                "naive_or_alarm"
            ),

        "hybrid_false_alarm_episodes":
            count_alarm_episodes(
                nominal_subset,
                "hybrid_alarm"
            ),

        # ====================================================
        # Global detection across all fault amplitudes
        # ====================================================

        "fdir_pd_global":
            100.0
            *
            np.mean(
                all_fault_subset[
                    "fdir_detected"
                ]
            ),

        "temporal_ai_pd_global":
            100.0
            *
            np.mean(
                all_fault_subset[
                    "ai_raw_alarm"
                ]
            ),

        "naive_or_pd_global":
            100.0
            *
            np.mean(
                all_fault_subset[
                    "naive_or_alarm"
                ]
            ),

        "hybrid_pd_global":
            100.0
            *
            np.mean(
                all_fault_subset[
                    "hybrid_alarm"
                ]
            )
    }


    # ========================================================
    # Per-amplitude statistics
    # ========================================================

    for fault_magnitude_m in [
        5.0,
        10.0,
        20.0,
        50.0
    ]:

        fault_subset = run_dataset[
            (
                run_dataset[
                    "fault_magnitude_m"
                ]
                ==
                fault_magnitude_m
            )
            &
            (
                run_dataset[
                    "fault_active"
                ]
                ==
                1
            )
        ]


        suffix = (
            str(
                int(
                    fault_magnitude_m
                )
            )
        )


        row[
            f"fdir_pd_{suffix}"
        ] = (
            100.0
            *
            np.mean(
                fault_subset[
                    "fdir_detected"
                ]
            )
        )


        row[
            f"temporal_ai_pd_{suffix}"
        ] = (
            100.0
            *
            np.mean(
                fault_subset[
                    "ai_raw_alarm"
                ]
            )
        )


        row[
            f"naive_or_pd_{suffix}"
        ] = (
            100.0
            *
            np.mean(
                fault_subset[
                    "naive_or_alarm"
                ]
            )
        )


        row[
            f"hybrid_pd_{suffix}"
        ] = (
            100.0
            *
            np.mean(
                fault_subset[
                    "hybrid_alarm"
                ]
            )
        )


        # ====================================================
        # Episode detection
        # ====================================================

        row[
            f"fdir_episode_detected_{suffix}"
        ] = int(
            np.any(
                fault_subset[
                    "fdir_detected"
                ].to_numpy(
                    dtype=bool
                )
            )
        )


        row[
            f"hybrid_episode_detected_{suffix}"
        ] = int(
            np.any(
                fault_subset[
                    "hybrid_alarm"
                ].to_numpy(
                    dtype=bool
                )
            )
        )


        # ====================================================
        # Conditional latency
        # ====================================================

        row[
            f"fdir_latency_{suffix}_s"
        ] = (
            compute_first_alarm_latency(
                fault_subset,
                "fdir_detected"
            )
        )


        row[
            f"hybrid_latency_{suffix}_s"
        ] = (
            compute_first_alarm_latency(
                fault_subset,
                "hybrid_alarm"
            )
        )


    run_metric_rows.append(
        row
    )


run_metrics = pd.DataFrame(
    run_metric_rows
)


# ------------------------------------------------------------
# 20. CONFIDENCE INTERVAL HELPER
# ------------------------------------------------------------

def mean_std_ci95(
    values
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


    number = len(
        values
    )


    mean_value = np.mean(
        values
    )


    if number < 2:

        return (
            mean_value,
            np.nan,
            np.nan,
            np.nan
        )


    std_value = np.std(
        values,
        ddof=1
    )


    standard_error = (
        std_value
        /
        np.sqrt(
            number
        )
    )


    critical_value = student_t.ppf(
        0.975,
        df=
            number
            -
            1
    )


    lower = (
        mean_value
        -
        critical_value
        *
        standard_error
    )


    upper = (
        mean_value
        +
        critical_value
        *
        standard_error
    )


    return (
        mean_value,
        std_value,
        lower,
        upper
    )


# ------------------------------------------------------------
# 21. GLOBAL PAIRED DETECTION BENEFIT
# ------------------------------------------------------------

global_detection_difference = (
    run_metrics[
        "hybrid_pd_global"
    ]
    -
    run_metrics[
        "fdir_pd_global"
    ]
)


(
    global_gain_mean,
    global_gain_std,
    global_gain_lower,
    global_gain_upper
) = mean_std_ci95(
    global_detection_difference
)


# ------------------------------------------------------------
# 22. FALSE ALARM ENSEMBLE
# ------------------------------------------------------------

(
    hybrid_pfa_mean,
    hybrid_pfa_std,
    hybrid_pfa_lower,
    hybrid_pfa_upper
) = mean_std_ci95(
    run_metrics[
        "hybrid_pfa"
    ]
)


(
    fdir_pfa_mean,
    fdir_pfa_std,
    _,
    _
) = mean_std_ci95(
    run_metrics[
        "fdir_pfa"
    ]
)


(
    temporal_pfa_mean,
    temporal_pfa_std,
    _,
    _
) = mean_std_ci95(
    run_metrics[
        "temporal_ai_pfa"
    ]
)


# ------------------------------------------------------------
# 23. AMPLITUDE SUMMARY
# ------------------------------------------------------------

amplitude_summary_rows = []


for fault_magnitude_m in [
    5.0,
    10.0,
    20.0,
    50.0
]:

    suffix = str(
        int(
            fault_magnitude_m
        )
    )


    fdir_values = (
        run_metrics[
            f"fdir_pd_{suffix}"
        ].to_numpy()
    )


    temporal_values = (
        run_metrics[
            f"temporal_ai_pd_{suffix}"
        ].to_numpy()
    )


    hybrid_values = (
        run_metrics[
            f"hybrid_pd_{suffix}"
        ].to_numpy()
    )


    difference_values = (
        hybrid_values
        -
        fdir_values
    )


    (
        fdir_mean,
        fdir_std,
        _,
        _
    ) = mean_std_ci95(
        fdir_values
    )


    (
        temporal_mean,
        temporal_std,
        _,
        _
    ) = mean_std_ci95(
        temporal_values
    )


    (
        hybrid_mean,
        hybrid_std,
        _,
        _
    ) = mean_std_ci95(
        hybrid_values
    )


    (
        difference_mean,
        difference_std,
        difference_lower,
        difference_upper
    ) = mean_std_ci95(
        difference_values
    )


    fdir_episode_detection_rate = (
        100.0
        *
        np.mean(
            run_metrics[
                f"fdir_episode_detected_{suffix}"
            ]
        )
    )


    hybrid_episode_detection_rate = (
        100.0
        *
        np.mean(
            run_metrics[
                f"hybrid_episode_detected_{suffix}"
            ]
        )
    )


    fdir_latencies = (
        run_metrics[
            f"fdir_latency_{suffix}_s"
        ].to_numpy(
            dtype=float
        )
    )


    hybrid_latencies = (
        run_metrics[
            f"hybrid_latency_{suffix}_s"
        ].to_numpy(
            dtype=float
        )
    )


    valid_fdir_latencies = (
        fdir_latencies[
            np.isfinite(
                fdir_latencies
            )
        ]
    )


    valid_hybrid_latencies = (
        hybrid_latencies[
            np.isfinite(
                hybrid_latencies
            )
        ]
    )


    if len(
        valid_fdir_latencies
    ) > 0:

        fdir_mean_latency = np.mean(
            valid_fdir_latencies
        )

    else:

        fdir_mean_latency = (
            np.nan
        )


    if len(
        valid_hybrid_latencies
    ) > 0:

        hybrid_mean_latency = np.mean(
            valid_hybrid_latencies
        )

    else:

        hybrid_mean_latency = (
            np.nan
        )


    amplitude_summary_rows.append(
        {
            "fault_magnitude_m":
                fault_magnitude_m,

            "fdir_pd_mean":
                fdir_mean,

            "fdir_pd_std":
                fdir_std,

            "temporal_pd_mean":
                temporal_mean,

            "temporal_pd_std":
                temporal_std,

            "hybrid_pd_mean":
                hybrid_mean,

            "hybrid_pd_std":
                hybrid_std,

            "hybrid_minus_fdir_mean":
                difference_mean,

            "hybrid_minus_fdir_ci95_lower":
                difference_lower,

            "hybrid_minus_fdir_ci95_upper":
                difference_upper,

            "fdir_episode_detection_rate":
                fdir_episode_detection_rate,

            "hybrid_episode_detection_rate":
                hybrid_episode_detection_rate,

            "fdir_mean_latency_s":
                fdir_mean_latency,

            "hybrid_mean_latency_s":
                hybrid_mean_latency
        }
    )


amplitude_summary = pd.DataFrame(
    amplitude_summary_rows
)


# ------------------------------------------------------------
# 24. GLOBAL DETECTION MEANS
# ------------------------------------------------------------

(
    fdir_global_pd_mean,
    fdir_global_pd_std,
    _,
    _
) = mean_std_ci95(
    run_metrics[
        "fdir_pd_global"
    ]
)


(
    temporal_global_pd_mean,
    temporal_global_pd_std,
    _,
    _
) = mean_std_ci95(
    run_metrics[
        "temporal_ai_pd_global"
    ]
)


(
    hybrid_global_pd_mean,
    hybrid_global_pd_std,
    _,
    _
) = mean_std_ci95(
    run_metrics[
        "hybrid_pd_global"
    ]
)


# ------------------------------------------------------------
# 25. FALSE ALARM EPISODES
# ------------------------------------------------------------

mean_fdir_false_alarm_episodes = np.mean(
    run_metrics[
        "fdir_false_alarm_episodes"
    ]
)


mean_temporal_false_alarm_episodes = np.mean(
    run_metrics[
        "temporal_ai_false_alarm_episodes"
    ]
)


mean_naive_false_alarm_episodes = np.mean(
    run_metrics[
        "naive_or_false_alarm_episodes"
    ]
)


mean_hybrid_false_alarm_episodes = np.mean(
    run_metrics[
        "hybrid_false_alarm_episodes"
    ]
)


# ------------------------------------------------------------
# 26. IMPORTANT AMPLITUDE ROWS
# ------------------------------------------------------------

row_20m = (
    amplitude_summary[
        amplitude_summary[
            "fault_magnitude_m"
        ]
        ==
        20.0
    ]
    .iloc[
        0
    ]
)


row_50m = (
    amplitude_summary[
        amplitude_summary[
            "fault_magnitude_m"
        ]
        ==
        50.0
    ]
    .iloc[
        0
    ]
)


# ------------------------------------------------------------
# 27. VALIDATION CONDITIONS
# ------------------------------------------------------------

dataset_has_all_runs = (
    valid_dataset[
        "run_id"
    ].nunique()
    ==
    number_of_test_runs
)


methodology_valid = (
    heldout_runs_disjoint
    and
    all_temporal_features_finite
    and
    dataset_has_all_runs
)


hybrid_false_alarm_valid = (
    hybrid_pfa_mean
    <=
    5.0
)


global_gain_statistically_positive = (
    global_gain_lower
    >
    0.0
)


gain_20m_statistically_positive = (
    row_20m[
        "hybrid_minus_fdir_ci95_lower"
    ]
    >
    0.0
)


gain_50m_statistically_positive = (
    row_50m[
        "hybrid_minus_fdir_ci95_lower"
    ]
    >
    0.0
)


fifty_meter_episode_detection_valid = (
    row_50m[
        "hybrid_episode_detection_rate"
    ]
    >=
    95.0
)


validation_015e = (
    methodology_valid
    and
    hybrid_false_alarm_valid
    and
    global_gain_statistically_positive
    and
    gain_20m_statistically_positive
    and
    gain_50m_statistically_positive
    and
    fifty_meter_episode_detection_valid
)


# ------------------------------------------------------------
# 28. SAVE
# ------------------------------------------------------------

valid_dataset.to_csv(
    heldout_dataset_path,
    index=False
)


run_metrics.to_csv(
    run_metrics_path,
    index=False
)


amplitude_summary.to_csv(
    amplitude_summary_path,
    index=False
)


# ------------------------------------------------------------
# 29. PRINT RESULTS
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-E RESULTS"
)

print(
    "Fresh held-out Monte Carlo validation"
)

print(
    "========================================================================================================================"
)

print(
    f"Held-out Monte Carlo runs : "
    f"{number_of_test_runs}"
)

print(
    f"Raw rows : "
    f"{len(heldout_dataset)}"
)

print(
    f"Rows with complete temporal history : "
    f"{len(valid_dataset)}"
)

print(
    f"All temporal features finite : "
    f"{all_temporal_features_finite}"
)

print(
    f"All held-out runs present : "
    f"{dataset_has_all_runs}"
)

print(
    f"Train/calibration/test disjoint : "
    f"{heldout_runs_disjoint}"
)

print(
    f"Time synchronization error : "
    f"{maximum_time_error:.6e} s"
)

print()


# ------------------------------------------------------------
# 30. GLOBAL PERFORMANCE
# ------------------------------------------------------------

print(
    "----- GLOBAL HELD-OUT PERFORMANCE -----"
)

print(
    f"Classical FDIR P_D : "
    f"{fdir_global_pd_mean:.2f} +/- "
    f"{fdir_global_pd_std:.2f} %"
)

print(
    f"Temporal AI P_D : "
    f"{temporal_global_pd_mean:.2f} +/- "
    f"{temporal_global_pd_std:.2f} %"
)

print(
    f"Persistence hybrid P_D : "
    f"{hybrid_global_pd_mean:.2f} +/- "
    f"{hybrid_global_pd_std:.2f} %"
)

print()

print(
    f"Paired gain hybrid - FDIR : "
    f"{global_gain_mean:.2f} percentage points"
)

print(
    f"95% CI gain : "
    f"[{global_gain_lower:.2f}, "
    f"{global_gain_upper:.2f}] percentage points"
)

print()


# ------------------------------------------------------------
# 31. FALSE ALARMS
# ------------------------------------------------------------

print(
    "----- HELD-OUT FALSE ALARMS -----"
)

print(
    f"Classical FDIR P_FA : "
    f"{fdir_pfa_mean:.2f} +/- "
    f"{fdir_pfa_std:.2f} %"
)

print(
    f"Temporal AI P_FA : "
    f"{temporal_pfa_mean:.2f} +/- "
    f"{temporal_pfa_std:.2f} %"
)

print(
    f"Persistence hybrid P_FA : "
    f"{hybrid_pfa_mean:.2f} +/- "
    f"{hybrid_pfa_std:.2f} %"
)

print(
    f"Hybrid P_FA 95% CI : "
    f"[{hybrid_pfa_lower:.2f}, "
    f"{hybrid_pfa_upper:.2f}] %"
)

print()

print(
    f"Mean false-alarm episodes/run — FDIR : "
    f"{mean_fdir_false_alarm_episodes:.2f}"
)

print(
    f"Mean false-alarm episodes/run — Temporal AI : "
    f"{mean_temporal_false_alarm_episodes:.2f}"
)

print(
    f"Mean false-alarm episodes/run — Naive OR : "
    f"{mean_naive_false_alarm_episodes:.2f}"
)

print(
    f"Mean false-alarm episodes/run — Persistence hybrid : "
    f"{mean_hybrid_false_alarm_episodes:.2f}"
)

print()


# ------------------------------------------------------------
# 32. AMPLITUDE TABLE
# ------------------------------------------------------------

print(
    "----- PERFORMANCE PAR AMPLITUDE -----"
)


print(
    amplitude_summary.to_string(
        index=False,
        formatters={
            "fault_magnitude_m":
                lambda value:
                    f"{value:6.1f}",

            "fdir_pd_mean":
                lambda value:
                    f"{value:7.2f}",

            "fdir_pd_std":
                lambda value:
                    f"{value:6.2f}",

            "temporal_pd_mean":
                lambda value:
                    f"{value:7.2f}",

            "temporal_pd_std":
                lambda value:
                    f"{value:6.2f}",

            "hybrid_pd_mean":
                lambda value:
                    f"{value:7.2f}",

            "hybrid_pd_std":
                lambda value:
                    f"{value:6.2f}",

            "hybrid_minus_fdir_mean":
                lambda value:
                    f"{value:+7.2f}",

            "hybrid_minus_fdir_ci95_lower":
                lambda value:
                    f"{value:+7.2f}",

            "hybrid_minus_fdir_ci95_upper":
                lambda value:
                    f"{value:+7.2f}",

            "fdir_episode_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "hybrid_episode_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "fdir_mean_latency_s":
                lambda value:
                    f"{value:7.1f}",

            "hybrid_mean_latency_s":
                lambda value:
                    f"{value:7.1f}"
        }
    )
)

print()


# ------------------------------------------------------------
# 33. KEY SCIENTIFIC RESULTS
# ------------------------------------------------------------

print(
    "----- KEY SCIENTIFIC RESULTS -----"
)

print(
    f"+20 m hybrid gain CI entirely > 0 : "
    f"{gain_20m_statistically_positive}"
)

print(
    f"+50 m hybrid gain CI entirely > 0 : "
    f"{gain_50m_statistically_positive}"
)

print(
    f"+50 m hybrid episode detection : "
    f"{row_50m['hybrid_episode_detection_rate']:.2f} %"
)

print()


# ------------------------------------------------------------
# 34. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)

print(
    f"Methodology valid : "
    f"{methodology_valid}"
)

print(
    f"Hybrid mean P_FA <= 5% : "
    f"{hybrid_false_alarm_valid}"
)

print(
    f"Global hybrid gain statistically positive : "
    f"{global_gain_statistically_positive}"
)

print(
    f"+20 m hybrid gain statistically positive : "
    f"{gain_20m_statistically_positive}"
)

print(
    f"+50 m hybrid gain statistically positive : "
    f"{gain_50m_statistically_positive}"
)

print(
    f"+50 m episode detection >= 95% : "
    f"{fifty_meter_episode_detection_valid}"
)

print()

print(
    f"VALIDATION GLOBALE 015-E : "
    f"{validation_015e}"
)

print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 35. FIGURE — DETECTION VS AMPLITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    amplitude_summary[
        "fault_magnitude_m"
    ],

    amplitude_summary[
        "fdir_pd_mean"
    ],

    yerr=
        amplitude_summary[
            "fdir_pd_std"
        ],

    marker="o",

    label="Classical FDIR"
)


plt.errorbar(
    amplitude_summary[
        "fault_magnitude_m"
    ],

    amplitude_summary[
        "temporal_pd_mean"
    ],

    yerr=
        amplitude_summary[
            "temporal_pd_std"
        ],

    marker="o",

    label="Temporal AI"
)


plt.errorbar(
    amplitude_summary[
        "fault_magnitude_m"
    ],

    amplitude_summary[
        "hybrid_pd_mean"
    ],

    yerr=
        amplitude_summary[
            "hybrid_pd_std"
        ],

    marker="o",

    label="Persistence hybrid"
)


plt.xlabel(
    "Persistent pseudorange fault [m]"
)

plt.ylabel(
    "Mean epoch detection rate [%]"
)

plt.title(
    "AURORA — 015-E held-out Monte Carlo detection"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase15_015e_detection_vs_fault.png",

    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 36. FIGURE — PAIRED +50 M PERFORMANCE
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    run_metrics[
        "fdir_pd_50"
    ],

    run_metrics[
        "hybrid_pd_50"
    ]
)


maximum_detection = (
    100.0
)


plt.plot(
    [
        0.0,
        maximum_detection
    ],
    [
        0.0,
        maximum_detection
    ],
    linestyle="--",
    label="Equal performance"
)


plt.xlabel(
    "Classical FDIR detection [%]"
)

plt.ylabel(
    "Hybrid detection [%]"
)

plt.title(
    "AURORA — Paired held-out +50 m detection"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase15_015e_paired_50m_detection.png",

    dpi=200
)

plt.show()