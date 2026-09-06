from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
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


# ============================================================
# AURORA
# Experiment 015-C
#
# TEMPORAL GNSS ANOMALY DETECTION
#
#
# Research question:
#
# Can causal temporal information improve GNSS anomaly
# detection under low measurement redundancy?
#
#
# Scenario:
#
#   100 - 120 min
#
#   exactly 5 satellites
#
#   persistent pseudorange fault:
#
#       105 - 115 min
#
#
# Fault amplitudes:
#
#       0
#       5
#       10
#       20
#       50 m
#
#
# Detectors:
#
#   1. Classical FDIR
#
#   2. Instantaneous Isolation Forest
#
#   3. Temporal Isolation Forest
#
#   4. Hybrid:
#
#          FDIR OR Temporal Isolation Forest
#
#
# Both AI models:
#
#   - trained ONLY on nominal sequences
#   - calibrated ONLY on independent nominal sequences
#   - evaluated on held-out noise runs
#
#
# Temporal features are strictly causal.
# No future information is used.
# ============================================================


# ------------------------------------------------------------
# 1. OUTPUT
# ------------------------------------------------------------

data_directory = (
    Path(
        "data"
    )
    /
    "phase15"
)

table_directory = (
    Path(
        "results"
    )
    /
    "tables"
)

figure_directory = (
    Path(
        "results"
    )
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


sequence_dataset_path = (
    data_directory
    /
    "gnss_temporal_dataset_015c.csv"
)


comparison_table_path = (
    table_directory
    /
    "phase15_015c_temporal_detection.csv"
)


# ------------------------------------------------------------
# 2. MONTE CARLO / SPLIT
# ------------------------------------------------------------

number_of_sequence_runs = (
    8
)


training_run_ids = [
    0,
    1,
    2
]


calibration_run_ids = [
    3
]


test_run_ids = [
    4,
    5,
    6,
    7
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
#
# Controlled approximation of the navigation prediction.
#
# The prior is NOT truth itself:
#
#   prior = truth + unknown fixed run-level offset
#
# Same prior is used for every fault amplitude.
# ------------------------------------------------------------

prior_position_sigma_m = (
    20.0
)


prior_clock_sigma_m = (
    10.0
)


# ------------------------------------------------------------
# 6. ORBIT TRUTH
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
# 7. HELPERS
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


def choose_fault_target(
    working_ids
):

    if (
        preferred_fault_satellite
        in
        working_ids
    ):

        return (
            preferred_fault_satellite
        )

    return (
        working_ids[
            0
        ]
    )


# ------------------------------------------------------------
# 8. GENERATE CONTINUOUS SEQUENCE DATASET
# ------------------------------------------------------------

rows = []


print(
    "\n"
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-C"
)

print(
    "Causal temporal anomaly detection"
)

print(
    "========================================================================================================================"
)

print(
    f"Sequence : "
    f"{sequence_start_minutes:.1f}"
    f" - "
    f"{sequence_end_minutes:.1f} min"
)

print(
    f"Persistent fault : "
    f"{fault_start_minutes:.1f}"
    f" - "
    f"{fault_end_minutes:.1f} min"
)

print(
    f"Runs : "
    f"{number_of_sequence_runs}"
)

print(
    f"Fault amplitudes : "
    f"{fault_magnitudes_m}"
)

print(
    "Generating continuous paired sequences..."
)


for run_id in range(
    number_of_sequence_runs
):

    pseudorange_rng = np.random.default_rng(
        50_000
        +
        run_id
    )


    prior_rng = np.random.default_rng(
        60_000
        +
        run_id
    )


    # ========================================================
    # Fixed run-level navigation prediction error
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
        # Visible GNSS satellites
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
        # One common pseudorange-noise realization
        # for ALL fault amplitudes.
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
            # FDIR
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
            # Full solution
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
            # Instantaneous features
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
        f"{run_id + 1:02d}/"
        f"{number_of_sequence_runs} : "
        f"{len(rows) - rows_before} rows"
    )


# ------------------------------------------------------------
# 9. DATAFRAME
# ------------------------------------------------------------

dataset = pd.DataFrame(
    rows
)


if len(
    dataset
) == 0:

    raise RuntimeError(
        "Le dataset temporel est vide."
    )


dataset = (
    add_causal_temporal_features(
        dataframe=
            dataset,

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
        dataset
    )
)


dataset[
    "temporal_features_valid"
] = (
    valid_temporal_mask.astype(
        int
    )
)


dataset.to_csv(
    sequence_dataset_path,
    index=False
)


# ------------------------------------------------------------
# 10. COMMON VALID DATASET
#
# Both instantaneous and temporal AI detectors are evaluated
# on the SAME epochs.
# ------------------------------------------------------------

common_dataset = dataset[
    valid_temporal_mask
].copy()


# ------------------------------------------------------------
# 11. INSTANTANEOUS FEATURE SET
# ------------------------------------------------------------

INSTANT_FEATURE_COLUMNS = [
    "number_of_satellites",
    "pdop",
    "chi2_ratio",
    "residual_rms_m",
    "residual_mean_abs_m",
    "residual_max_abs_m",
    "residual_peak_to_peak_m",
    "residual_peak_over_rms",
    "prior_position_correction_m",
    "prior_clock_correction_m"
]


# ------------------------------------------------------------
# 12. SPLITS
# ------------------------------------------------------------

training_nominal = common_dataset[
    (
        common_dataset[
            "run_id"
        ].isin(
            training_run_ids
        )
    )
    &
    (
        common_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


calibration_nominal = common_dataset[
    (
        common_dataset[
            "run_id"
        ].isin(
            calibration_run_ids
        )
    )
    &
    (
        common_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


# ============================================================
# Test nominal:
#
# only the 0 m sequence
# ============================================================

test_nominal = common_dataset[
    (
        common_dataset[
            "run_id"
        ].isin(
            test_run_ids
        )
    )
    &
    (
        common_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


# ============================================================
# Test fault:
#
# only epochs where the fault is actually active.
# ============================================================

test_fault = common_dataset[
    (
        common_dataset[
            "run_id"
        ].isin(
            test_run_ids
        )
    )
    &
    (
        common_dataset[
            "fault_active"
        ]
        ==
        1
    )
].copy()


test_dataset = pd.concat(
    (
        test_nominal,
        test_fault
    ),
    axis=0,
    ignore_index=True
)


test_labels = (
    test_dataset[
        "fault_active"
    ].to_numpy(
        dtype=int
    )
)


# ------------------------------------------------------------
# 13. VERIFY SPLITS
# ------------------------------------------------------------

training_runs_actual = set(
    training_nominal[
        "run_id"
    ].unique()
)


calibration_runs_actual = set(
    calibration_nominal[
        "run_id"
    ].unique()
)


test_runs_actual = set(
    test_dataset[
        "run_id"
    ].unique()
)


splits_disjoint = (
    len(
        training_runs_actual.intersection(
            calibration_runs_actual
        )
    )
    ==
    0
    and
    len(
        training_runs_actual.intersection(
            test_runs_actual
        )
    )
    ==
    0
    and
    len(
        calibration_runs_actual.intersection(
            test_runs_actual
        )
    )
    ==
    0
)


# ------------------------------------------------------------
# 14. INSTANTANEOUS ISOLATION FOREST
# ------------------------------------------------------------

instant_detector = (
    IsolationForestAnomalyDetector(
        feature_columns=
            INSTANT_FEATURE_COLUMNS,

        n_estimators=
            500,

        random_state=
            1510
    )
)


instant_detector.fit_nominal(
    training_nominal
)


instant_calibration = (
    instant_detector.calibrate_threshold(
        nominal_calibration_dataframe=
            calibration_nominal,

        target_false_alarm_rate=
            0.01
    )
)


instant_scores = (
    instant_detector.score_samples(
        test_dataset
    )
)


instant_predictions = (
    instant_detector.predict(
        test_dataset
    )
)


# ------------------------------------------------------------
# 15. TEMPORAL ISOLATION FOREST
# ------------------------------------------------------------

temporal_detector = (
    IsolationForestAnomalyDetector(
        feature_columns=
            TEMPORAL_FEATURE_COLUMNS,

        n_estimators=
            500,

        random_state=
            1511
    )
)


temporal_detector.fit_nominal(
    training_nominal
)


temporal_calibration = (
    temporal_detector.calibrate_threshold(
        nominal_calibration_dataframe=
            calibration_nominal,

        target_false_alarm_rate=
            0.01
    )
)


temporal_scores = (
    temporal_detector.score_samples(
        test_dataset
    )
)


temporal_predictions = (
    temporal_detector.predict(
        test_dataset
    )
)


# ------------------------------------------------------------
# 16. CLASSICAL FDIR + HYBRID
# ------------------------------------------------------------

fdir_predictions = (
    test_dataset[
        "fdir_detected"
    ].to_numpy(
        dtype=int
    )
)


hybrid_predictions = np.logical_or(
    fdir_predictions
    ==
    1,

    temporal_predictions
    ==
    1
).astype(
    int
)


# ------------------------------------------------------------
# 17. METRICS
# ------------------------------------------------------------

def compute_binary_metrics(
    labels,
    predictions
):

    labels = np.asarray(
        labels,
        dtype=int
    )

    predictions = np.asarray(
        predictions,
        dtype=int
    )


    nominal_mask = (
        labels
        ==
        0
    )


    fault_mask = (
        labels
        ==
        1
    )


    return {
        "false_alarm_rate":
            100.0
            *
            np.mean(
                predictions[
                    nominal_mask
                ]
            ),

        "detection_rate":
            100.0
            *
            np.mean(
                predictions[
                    fault_mask
                ]
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
    }


fdir_metrics = (
    compute_binary_metrics(
        test_labels,
        fdir_predictions
    )
)


instant_metrics = (
    compute_binary_metrics(
        test_labels,
        instant_predictions
    )
)


temporal_metrics = (
    compute_binary_metrics(
        test_labels,
        temporal_predictions
    )
)


hybrid_metrics = (
    compute_binary_metrics(
        test_labels,
        hybrid_predictions
    )
)


# ------------------------------------------------------------
# 18. CONTINUOUS SCORE METRICS
# ------------------------------------------------------------

instant_roc_auc = (
    roc_auc_score(
        test_labels,
        instant_scores
    )
)


temporal_roc_auc = (
    roc_auc_score(
        test_labels,
        temporal_scores
    )
)


instant_average_precision = (
    average_precision_score(
        test_labels,
        instant_scores
    )
)


temporal_average_precision = (
    average_precision_score(
        test_labels,
        temporal_scores
    )
)


# ------------------------------------------------------------
# 19. SAVE PREDICTIONS
# ------------------------------------------------------------

test_dataset[
    "instant_score"
] = (
    instant_scores
)


test_dataset[
    "temporal_score"
] = (
    temporal_scores
)


test_dataset[
    "instant_detected"
] = (
    instant_predictions
)


test_dataset[
    "temporal_detected"
] = (
    temporal_predictions
)


test_dataset[
    "hybrid_detected"
] = (
    hybrid_predictions
)


# ------------------------------------------------------------
# 20. PERFORMANCE BY FAULT AMPLITUDE
# ------------------------------------------------------------

comparison_rows = []


for fault_magnitude_m in [
    5.0,
    10.0,
    20.0,
    50.0
]:

    subset = test_dataset[
        (
            test_dataset[
                "fault_magnitude_m"
            ]
            ==
            fault_magnitude_m
        )
        &
        (
            test_dataset[
                "fault_active"
            ]
            ==
            1
        )
    ]


    if len(
        subset
    ) == 0:

        continue


    comparison_rows.append(
        {
            "fault_magnitude_m":
                fault_magnitude_m,

            "samples":
                len(
                    subset
                ),

            "fdir_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "fdir_detected"
                    ]
                ),

            "instant_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "instant_detected"
                    ]
                ),

            "temporal_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "temporal_detected"
                    ]
                ),

            "hybrid_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "hybrid_detected"
                    ]
                ),

            "mean_instant_score":
                np.mean(
                    subset[
                        "instant_score"
                    ]
                ),

            "mean_temporal_score":
                np.mean(
                    subset[
                        "temporal_score"
                    ]
                )
        }
    )


comparison_table = pd.DataFrame(
    comparison_rows
)


comparison_table.to_csv(
    comparison_table_path,
    index=False
)


# ------------------------------------------------------------
# 21. DETECTION LATENCY
#
# First alarm after start of persistent fault.
#
# Calculated per run and fault magnitude.
# ------------------------------------------------------------

latency_rows = []


for run_id in test_run_ids:

    for fault_magnitude_m in [
        5.0,
        10.0,
        20.0,
        50.0
    ]:

        subset = common_dataset[
            (
                common_dataset[
                    "run_id"
                ]
                ==
                run_id
            )
            &
            (
                common_dataset[
                    "fault_magnitude_m"
                ]
                ==
                fault_magnitude_m
            )
            &
            (
                common_dataset[
                    "fault_active"
                ]
                ==
                1
            )
        ].copy()


        if len(
            subset
        ) == 0:

            continue


        instant_subset_scores = (
            instant_detector.score_samples(
                subset
            )
        )


        temporal_subset_scores = (
            temporal_detector.score_samples(
                subset
            )
        )


        instant_subset_predictions = (
            instant_subset_scores
            >=
            instant_detector.threshold
        )


        temporal_subset_predictions = (
            temporal_subset_scores
            >=
            temporal_detector.threshold
        )


        fdir_subset_predictions = (
            subset[
                "fdir_detected"
            ].to_numpy(
                dtype=bool
            )
        )


        hybrid_subset_predictions = (
            fdir_subset_predictions
            |
            temporal_subset_predictions
        )


        times = (
            subset[
                "time_seconds"
            ].to_numpy()
        )


        fault_start_seconds = (
            fault_start_minutes
            *
            60.0
        )


        def first_alarm_latency(
            predictions
        ):

            alarm_indices = np.where(
                predictions
            )[0]


            if len(
                alarm_indices
            ) == 0:

                return np.nan


            first_alarm_time = (
                times[
                    alarm_indices[
                        0
                    ]
                ]
            )


            return (
                first_alarm_time
                -
                fault_start_seconds
            )


        latency_rows.append(
            {
                "run_id":
                    run_id,

                "fault_magnitude_m":
                    fault_magnitude_m,

                "fdir_latency_s":
                    first_alarm_latency(
                        fdir_subset_predictions
                    ),

                "instant_latency_s":
                    first_alarm_latency(
                        instant_subset_predictions
                    ),

                "temporal_latency_s":
                    first_alarm_latency(
                        temporal_subset_predictions
                    ),

                "hybrid_latency_s":
                    first_alarm_latency(
                        hybrid_subset_predictions
                    )
            }
        )


latency_table = pd.DataFrame(
    latency_rows
)


# ------------------------------------------------------------
# 22. LATENCY SUMMARY
# ------------------------------------------------------------

latency_summary_rows = []


for fault_magnitude_m in [
    5.0,
    10.0,
    20.0,
    50.0
]:

    subset = latency_table[
        latency_table[
            "fault_magnitude_m"
        ]
        ==
        fault_magnitude_m
    ]


    row = {
        "fault_magnitude_m":
            fault_magnitude_m
    }


    for detector_name in [
        "fdir",
        "instant",
        "temporal",
        "hybrid"
    ]:

        values = subset[
            f"{detector_name}_latency_s"
        ].to_numpy()


        finite_values = values[
            np.isfinite(
                values
            )
        ]


        if len(
            finite_values
        ) > 0:

            mean_latency = np.mean(
                finite_values
            )


            detected_episode_rate = (
                100.0
                *
                len(
                    finite_values
                )
                /
                len(
                    values
                )
            )

        else:

            mean_latency = (
                np.nan
            )


            detected_episode_rate = (
                0.0
            )


        row[
            f"{detector_name}_mean_latency_s"
        ] = (
            mean_latency
        )


        row[
            f"{detector_name}_episode_detection_rate"
        ] = (
            detected_episode_rate
        )


    latency_summary_rows.append(
        row
    )


latency_summary = pd.DataFrame(
    latency_summary_rows
)


# ------------------------------------------------------------
# 23. SCIENTIFIC FLAGS
# ------------------------------------------------------------

temporal_improves_global_detection = (
    temporal_metrics[
        "detection_rate"
    ]
    >
    instant_metrics[
        "detection_rate"
    ]
)


temporal_improves_50m = bool(
    comparison_table.loc[
        comparison_table[
            "fault_magnitude_m"
        ]
        ==
        50.0,
        "temporal_detection_rate"
    ].iloc[
        0
    ]
    >
    comparison_table.loc[
        comparison_table[
            "fault_magnitude_m"
        ]
        ==
        50.0,
        "instant_detection_rate"
    ].iloc[
        0
    ]
)


hybrid_improves_fdir = (
    hybrid_metrics[
        "detection_rate"
    ]
    >
    fdir_metrics[
        "detection_rate"
    ]
)


# ------------------------------------------------------------
# 24. VALIDATION
# ------------------------------------------------------------

methodology_valid = (
    splits_disjoint
    and
    len(
        training_nominal
    )
    >
    100
    and
    len(
        calibration_nominal
    )
    >
    30
)


temporal_false_alarm_valid = (
    temporal_metrics[
        "false_alarm_rate"
    ]
    <=
    5.0
)


temporal_score_valid = (
    temporal_roc_auc
    >
    0.60
)


hybrid_false_alarm_valid = (
    hybrid_metrics[
        "false_alarm_rate"
    ]
    <=
    6.0
)


validation_015c = (
    methodology_valid
    and
    temporal_false_alarm_valid
    and
    temporal_score_valid
    and
    hybrid_false_alarm_valid
)


# ------------------------------------------------------------
# 25. PRINT
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-C RESULTS"
)

print(
    "========================================================================================================================"
)

print(
    f"Raw temporal rows : "
    f"{len(dataset)}"
)

print(
    f"Rows with complete causal history : "
    f"{len(common_dataset)}"
)

print(
    f"Temporal feature count : "
    f"{len(TEMPORAL_FEATURE_COLUMNS)}"
)

print(
    f"Train nominal rows : "
    f"{len(training_nominal)}"
)

print(
    f"Calibration nominal rows : "
    f"{len(calibration_nominal)}"
)

print(
    f"Test nominal rows : "
    f"{len(test_nominal)}"
)

print(
    f"Test fault rows : "
    f"{len(test_fault)}"
)

print(
    f"Splits disjoint : "
    f"{splits_disjoint}"
)

print()


print(
    "----- THRESHOLDS -----"
)

print(
    f"Instant threshold : "
    f"{instant_calibration['threshold']:.6f}"
)

print(
    f"Instant calibration P_FA : "
    f"{100.0 * instant_calibration['achieved_false_alarm_rate']:.2f} %"
)

print(
    f"Temporal threshold : "
    f"{temporal_calibration['threshold']:.6f}"
)

print(
    f"Temporal calibration P_FA : "
    f"{100.0 * temporal_calibration['achieved_false_alarm_rate']:.2f} %"
)

print()


print(
    "----- GLOBAL TEST PERFORMANCE -----"
)

header = (
    f"{'Detector':>22} | "
    f"{'P_FA [%]':>10} | "
    f"{'P_D [%]':>10} | "
    f"{'Precision':>10} | "
    f"{'Recall':>10} | "
    f"{'F1':>10}"
)

print(
    header
)

print(
    "-" * len(
        header
    )
)


detector_metrics = [
    (
        "Classical FDIR",
        fdir_metrics
    ),
    (
        "Instant IF",
        instant_metrics
    ),
    (
        "Temporal IF",
        temporal_metrics
    ),
    (
        "FDIR OR Temporal",
        hybrid_metrics
    )
]


for detector_name, metrics in detector_metrics:

    print(
        f"{detector_name:>22} | "
        f"{metrics['false_alarm_rate']:10.2f} | "
        f"{metrics['detection_rate']:10.2f} | "
        f"{metrics['precision']:10.3f} | "
        f"{metrics['recall']:10.3f} | "
        f"{metrics['f1']:10.3f}"
    )


print()

print(
    f"Instant ROC-AUC : "
    f"{instant_roc_auc:.4f}"
)

print(
    f"Temporal ROC-AUC : "
    f"{temporal_roc_auc:.4f}"
)

print(
    f"Instant Average Precision : "
    f"{instant_average_precision:.4f}"
)

print(
    f"Temporal Average Precision : "
    f"{temporal_average_precision:.4f}"
)

print()


print(
    "----- DETECTION PAR AMPLITUDE -----"
)

print(
    comparison_table.to_string(
        index=False,
        formatters={
            "fault_magnitude_m":
                lambda value:
                    f"{value:6.1f}",

            "fdir_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "instant_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "temporal_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "hybrid_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "mean_instant_score":
                lambda value:
                    f"{value:.6f}",

            "mean_temporal_score":
                lambda value:
                    f"{value:.6f}"
        }
    )
)

print()


print(
    "----- DETECTION LATENCY -----"
)

print(
    latency_summary.to_string(
        index=False,
        formatters={
            column:
                (
                    lambda value:
                        f"{value:.1f}"
                )
            for column in latency_summary.columns
            if column
            !=
            "fault_magnitude_m"
        }
    )
)

print()


print(
    "----- SCIENTIFIC FLAGS -----"
)

print(
    f"Temporal improves global detection vs instant : "
    f"{temporal_improves_global_detection}"
)

print(
    f"Temporal improves +50 m vs instant : "
    f"{temporal_improves_50m}"
)

print(
    f"Hybrid improves classical FDIR : "
    f"{hybrid_improves_fdir}"
)

print()


print(
    "----- VALIDATION -----"
)

print(
    f"Methodology valid : "
    f"{methodology_valid}"
)

print(
    f"Temporal P_FA <= 5% : "
    f"{temporal_false_alarm_valid}"
)

print(
    f"Temporal ROC-AUC > 0.60 : "
    f"{temporal_score_valid}"
)

print(
    f"Hybrid P_FA <= 6% : "
    f"{hybrid_false_alarm_valid}"
)

print()

print(
    f"VALIDATION GLOBALE 015-C : "
    f"{validation_015c}"
)

print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 26. FIGURE — DETECTION RATE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    comparison_table[
        "fault_magnitude_m"
    ],
    comparison_table[
        "fdir_detection_rate"
    ],
    marker="o",
    label="Classical FDIR"
)


plt.plot(
    comparison_table[
        "fault_magnitude_m"
    ],
    comparison_table[
        "instant_detection_rate"
    ],
    marker="o",
    label="Instant IF"
)


plt.plot(
    comparison_table[
        "fault_magnitude_m"
    ],
    comparison_table[
        "temporal_detection_rate"
    ],
    marker="o",
    label="Temporal IF"
)


plt.plot(
    comparison_table[
        "fault_magnitude_m"
    ],
    comparison_table[
        "hybrid_detection_rate"
    ],
    marker="o",
    label="FDIR OR Temporal"
)


plt.xlabel(
    "Persistent pseudorange fault [m]"
)

plt.ylabel(
    "Detection rate [%]"
)

plt.title(
    "AURORA — Temporal anomaly detection under 5-satellite redundancy"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


detection_figure_path = (
    figure_directory
    /
    "phase15_015c_temporal_detection.png"
)


plt.savefig(
    detection_figure_path,
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 27. FIGURE — EXAMPLE TEMPORAL SCORE
#
# Example:
# test run 4, +20 m persistent fault.
# ------------------------------------------------------------

example_sequence = common_dataset[
    (
        common_dataset[
            "run_id"
        ]
        ==
        test_run_ids[
            0
        ]
    )
    &
    (
        common_dataset[
            "fault_magnitude_m"
        ]
        ==
        20.0
    )
].copy()


example_temporal_scores = (
    temporal_detector.score_samples(
        example_sequence
    )
)


example_instant_scores = (
    instant_detector.score_samples(
        example_sequence
    )
)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    example_sequence[
        "time_minutes"
    ],
    example_instant_scores,
    label="Instant anomaly score"
)


plt.plot(
    example_sequence[
        "time_minutes"
    ],
    example_temporal_scores,
    label="Temporal anomaly score"
)


plt.axhline(
    instant_detector.threshold,
    linestyle="--",
    label="Instant threshold"
)


plt.axhline(
    temporal_detector.threshold,
    linestyle=":",
    label="Temporal threshold"
)


plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15,
    label="+20 m persistent fault"
)


plt.xlabel(
    "Time [min]"
)

plt.ylabel(
    "Isolation Forest anomaly score"
)

plt.title(
    "AURORA — Example causal temporal anomaly score"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


score_figure_path = (
    figure_directory
    /
    "phase15_015c_temporal_score_example.png"
)


plt.savefig(
    score_figure_path,
    dpi=200
)

plt.show()