from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    AI_FEATURE_COLUMNS,
    extract_gnss_anomaly_features
)


# ============================================================
# AURORA
# Experience 015-A
#
# AI DATASET + FEATURE ENGINEERING
#
#
# Goal:
#
# Build a labelled dataset from the REAL AURORA GNSS/FDIR
# chain without leaking truth information into the features.
#
#
# Dataset dimensions:
#
#   several orbital geometries
#   several noise realizations
#
#   two redundancy regimes:
#
#       FULL
#       FIVE_SAT
#
#   fault amplitudes:
#
#       0 m
#       5 m
#       10 m
#       20 m
#       50 m
#
#
# Same pseudorange noise is reused across fault amplitudes
# inside a group.
#
# This gives a PAIRED dataset:
#
#       same geometry
#       same noise
#       only fault amplitude changes
#
#
# The label is ONLY for training/evaluation.
# It is NOT part of AI_FEATURE_COLUMNS.
# ============================================================


# ------------------------------------------------------------
# 1. REPRODUCIBILITY
# ------------------------------------------------------------

master_seed = (
    15001
)

number_of_noise_runs = (
    6
)


# ------------------------------------------------------------
# 2. GNSS SETTINGS
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

fault_magnitudes_m = np.array([
    0.0,
    5.0,
    10.0,
    20.0,
    50.0
])


# ------------------------------------------------------------
# 3. PRIOR / PREDICTION SETTINGS
#
# This emulates an EKF navigation prediction provided
# before the GNSS LS update.
# ------------------------------------------------------------

prior_position_sigma_m = (
    20.0
)

prior_clock_sigma_m = (
    10.0
)


# ------------------------------------------------------------
# 4. ORBIT
# ------------------------------------------------------------

simulation_duration_minutes = (
    150.0
)

simulation_duration = (
    simulation_duration_minutes
    *
    60.0
)

dt = (
    10.0
)

number_of_epochs = (
    int(
        simulation_duration
        /
        dt
    )
    +
    1
)

time = np.linspace(
    0.0,
    simulation_duration,
    number_of_epochs
)

time_minutes = (
    time
    /
    60.0
)


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
            simulation_duration,

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
# 5. SELECT ORBITAL SAMPLE TIMES
#
# We want:
#
#  - broad orbital geometry coverage
#  - dense sampling of 105-115 min because this was the
#    difficult low-redundancy region identified in Phase 14.
# ------------------------------------------------------------

regular_sample_minutes = np.linspace(
    5.0,
    145.0,
    20
)


difficult_sample_minutes = np.arange(
    105.0,
    115.0,
    20.0 / 60.0
)


all_sample_minutes = np.unique(
    np.concatenate(
        (
            regular_sample_minutes,
            difficult_sample_minutes
        )
    )
)


sample_indices = np.unique(
    np.clip(
        np.round(
            all_sample_minutes
            *
            60.0
            /
            dt
        ).astype(
            int
        ),
        1,
        number_of_epochs - 1
    )
)


# ------------------------------------------------------------
# 6. OUTPUT DIRECTORIES
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


dataset_path = (
    data_directory
    /
    "gnss_anomaly_dataset_015a.csv"
)

summary_path = (
    table_directory
    /
    "phase15_015a_dataset_summary.csv"
)

feature_summary_path = (
    table_directory
    /
    "phase15_015a_feature_summary.csv"
)


# ------------------------------------------------------------
# 7. HELPER — SELECT FIVE SATELLITES
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
# 9. DATASET GENERATION
# ------------------------------------------------------------

rows = []

number_of_failed_groups = (
    0
)

number_of_attempted_groups = (
    0
)


print(
    "\n"
    "================================================================================================================"
)

print(
    "AURORA — Experience 015-A"
)

print(
    "GNSS anomaly dataset + feature engineering"
)

print(
    "================================================================================================================"
)

print(
    f"Noise runs : "
    f"{number_of_noise_runs}"
)

print(
    f"Orbital sample epochs : "
    f"{len(sample_indices)}"
)

print(
    f"Redundancy regimes : "
    f"FULL, FIVE_SAT"
)

print(
    f"Fault magnitudes : "
    f"{fault_magnitudes_m.tolist()} m"
)

print(
    "Generation dataset..."
)


for run_index in range(
    number_of_noise_runs
):

    pseudorange_rng = (
        np.random.default_rng(
            master_seed
            +
            10_000
            *
            run_index
            +
            1
        )
    )


    prior_rng = (
        np.random.default_rng(
            master_seed
            +
            10_000
            *
            run_index
            +
            2
        )
    )


    run_rows_before = len(
        rows
    )


    for sample_counter, epoch_index in enumerate(
        sample_indices
    ):

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


        # ====================================================
        # Visible GNSS constellation
        # ====================================================

        (
            all_satellite_positions,
            all_satellite_ids
        ) = (
            generate_simplified_gps_positions(
                time_seconds=
                    reception_time_seconds
            )
        )


        (
            _,
            visible_ids
        ) = (
            select_visible_gnss_satellites(
                receiver_position=
                    receiver_position_true,

                satellite_positions=
                    all_satellite_positions,

                satellite_ids=
                    all_satellite_ids
            )
        )


        visible_ids = [
            str(
                satellite_id
            ).strip()
            for satellite_id in visible_ids
        ]


        if len(
            visible_ids
        ) < 6:

            continue


        # ====================================================
        # Same prior used for FULL and FIVE_SAT pair
        # ====================================================

        prior_position = (
            receiver_position_true
            +
            prior_rng.normal(
                0.0,
                prior_position_sigma_m,
                size=3
            )
        )


        prior_clock_bias_meters = (
            true_clock_bias_meters
            +
            prior_rng.normal(
                0.0,
                prior_clock_sigma_m
            )
        )


        # ====================================================
        # Two redundancy regimes
        # ====================================================

        for redundancy_mode in [
            "FULL",
            "FIVE_SAT"
        ]:

            number_of_attempted_groups += (
                1
            )


            if redundancy_mode == "FULL":

                working_ids = (
                    visible_ids.copy()
                )

            else:

                working_ids = (
                    select_five_satellites(
                        visible_ids
                    )
                )


            if (
                working_ids is None
                or
                len(
                    working_ids
                ) < 5
            ):

                number_of_failed_groups += (
                    1
                )

                continue


            # ================================================
            # Generate ONE base pseudorange noise realization
            #
            # Every fault amplitude in this group shares
            # exactly these same nominal measurements.
            # ================================================

            nominal_pseudorange_result = (
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
                nominal_pseudorange_result[
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


            group_id = (
                f"R{run_index:02d}"
                f"_E{epoch_index:04d}"
                f"_{redundancy_mode}"
            )


            temporary_group_rows = []


            # ================================================
            # Paired fault amplitudes
            # ================================================

            for fault_magnitude_m in fault_magnitudes_m:

                pseudoranges = (
                    nominal_pseudoranges.copy()
                )


                fault_label = int(
                    fault_magnitude_m
                    >
                    0.0
                )


                if fault_label == 1:

                    pseudoranges[
                        fault_target_index
                    ] += (
                        fault_magnitude_m
                    )


                # ============================================
                # Real Phase 9-E FDIR
                # ============================================

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

                    temporary_group_rows = []

                    break


                # ============================================
                # Full solution needed for PDOP
                # ============================================

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

                    temporary_group_rows = []

                    break


                # ============================================
                # Geometry / PDOP
                # ============================================

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

                    temporary_group_rows = []

                    break


                # ============================================
                # AI features
                # ============================================

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

                    temporary_group_rows = []

                    break


                # ============================================
                # FDIR diagnostics
                #
                # These are saved for comparison later,
                # but NOT included in AI_FEATURE_COLUMNS.
                # ============================================

                fdir_detected = bool(
                    fdir_result.get(
                        "fault_detected",
                        False
                    )
                )


                fdir_reconfigured = bool(
                    fdir_result.get(
                        "reconfigured",
                        False
                    )
                )


                fdir_measurement_accepted = bool(
                    fdir_result.get(
                        "measurement_accepted",
                        False
                    )
                )


                isolated_id = (
                    fdir_result.get(
                        "isolated_id",
                        None
                    )
                )


                correct_isolation = (
                    (
                        isolated_id
                        is not None
                    )
                    and
                    (
                        str(
                            isolated_id
                        ).strip()
                        ==
                        str(
                            fault_target
                        ).strip()
                    )
                )


                row = {
                    # ----------------------------------------
                    # Grouping metadata
                    # ----------------------------------------

                    "group_id":
                        group_id,

                    "run_id":
                        run_index,

                    "epoch_index":
                        epoch_index,

                    "time_seconds":
                        reception_time_seconds,

                    "time_minutes":
                        reception_time_minutes,

                    "redundancy_mode":
                        redundancy_mode,

                    # ----------------------------------------
                    # Ground-truth LABEL metadata
                    #
                    # NEVER part of AI_FEATURE_COLUMNS.
                    # ----------------------------------------

                    "fault_label":
                        fault_label,

                    "fault_magnitude_m":
                        float(
                            fault_magnitude_m
                        ),

                    "fault_satellite_id":
                        (
                            fault_target
                            if fault_label == 1
                            else ""
                        ),

                    # ----------------------------------------
                    # Classical FDIR output
                    #
                    # Used for benchmarking, not base ML input.
                    # ----------------------------------------

                    "fdir_detected":
                        int(
                            fdir_detected
                        ),

                    "fdir_reconfigured":
                        int(
                            fdir_reconfigured
                        ),

                    "fdir_measurement_accepted":
                        int(
                            fdir_measurement_accepted
                        ),

                    "fdir_correct_isolation":
                        int(
                            correct_isolation
                            if fault_label == 1
                            else False
                        ),

                    # ----------------------------------------
                    # Features
                    # ----------------------------------------

                    **features
                }


                temporary_group_rows.append(
                    row
                )


            # ================================================
            # Group accepted ONLY if every amplitude exists.
            #
            # Prevents paired-data imbalance.
            # ================================================

            if (
                len(
                    temporary_group_rows
                )
                ==
                len(
                    fault_magnitudes_m
                )
            ):

                rows.extend(
                    temporary_group_rows
                )

            else:

                number_of_failed_groups += (
                    1
                )


    print(
        f"Run "
        f"{run_index + 1:02d}/"
        f"{number_of_noise_runs} : "
        f"{len(rows) - run_rows_before} rows ajoutees"
    )


# ------------------------------------------------------------
# 10. DATAFRAME
# ------------------------------------------------------------

dataset = pd.DataFrame(
    rows
)


if len(
    dataset
) == 0:

    raise RuntimeError(
        "Le dataset 015-A est vide."
    )


# ------------------------------------------------------------
# 11. FEATURE VALIDATION
# ------------------------------------------------------------

missing_feature_columns = [
    column
    for column in AI_FEATURE_COLUMNS
    if column
    not in dataset.columns
]


if len(
    missing_feature_columns
) > 0:

    raise RuntimeError(
        "Features manquantes : "
        +
        str(
            missing_feature_columns
        )
    )


feature_matrix = dataset[
    AI_FEATURE_COLUMNS
].to_numpy(
    dtype=float
)


all_features_finite = bool(
    np.all(
        np.isfinite(
            feature_matrix
        )
    )
)


labels_not_in_features = (
    "fault_label"
    not in AI_FEATURE_COLUMNS
    and
    "fault_magnitude_m"
    not in AI_FEATURE_COLUMNS
    and
    "fdir_detected"
    not in AI_FEATURE_COLUMNS
)


# ------------------------------------------------------------
# 12. PAIRED GROUP VALIDATION
# ------------------------------------------------------------

expected_samples_per_group = (
    len(
        fault_magnitudes_m
    )
)


group_sizes = (
    dataset.groupby(
        "group_id"
    ).size()
)


all_groups_complete = bool(
    np.all(
        group_sizes.to_numpy()
        ==
        expected_samples_per_group
    )
)


number_of_complete_groups = (
    dataset[
        "group_id"
    ].nunique()
)


# ------------------------------------------------------------
# 13. CLASS COUNTS
# ------------------------------------------------------------

number_nominal_samples = int(
    np.sum(
        dataset[
            "fault_label"
        ]
        ==
        0
    )
)

number_fault_samples = int(
    np.sum(
        dataset[
            "fault_label"
        ]
        ==
        1
    )
)


both_classes_present = (
    number_nominal_samples
    >
    0
    and
    number_fault_samples
    >
    0
)


# ------------------------------------------------------------
# 14. FDIR SUMMARY
# ------------------------------------------------------------

summary = (
    dataset.groupby(
        [
            "redundancy_mode",
            "fault_magnitude_m"
        ],
        as_index=False
    )
    .agg(
        samples=(
            "fault_label",
            "size"
        ),

        fdir_detection_rate=(
            "fdir_detected",
            "mean"
        ),

        fdir_acceptance_rate=(
            "fdir_measurement_accepted",
            "mean"
        ),

        mean_satellites=(
            "number_of_satellites",
            "mean"
        ),

        mean_pdop=(
            "pdop",
            "mean"
        ),

        mean_chi2_ratio=(
            "chi2_ratio",
            "mean"
        ),

        mean_residual_rms_m=(
            "residual_rms_m",
            "mean"
        ),

        mean_residual_max_abs_m=(
            "residual_max_abs_m",
            "mean"
        )
    )
)


summary[
    "fdir_detection_rate"
] *= (
    100.0
)


summary[
    "fdir_acceptance_rate"
] *= (
    100.0
)


# ------------------------------------------------------------
# 15. FEATURE SUMMARY BY CLASS
# ------------------------------------------------------------

feature_summary_rows = []


for feature_name in AI_FEATURE_COLUMNS:

    nominal_values = (
        dataset.loc[
            dataset[
                "fault_label"
            ]
            ==
            0,
            feature_name
        ].to_numpy()
    )


    fault_values = (
        dataset.loc[
            dataset[
                "fault_label"
            ]
            ==
            1,
            feature_name
        ].to_numpy()
    )


    nominal_mean = np.mean(
        nominal_values
    )


    nominal_std = np.std(
        nominal_values,
        ddof=1
    )


    fault_mean = np.mean(
        fault_values
    )


    fault_std = np.std(
        fault_values,
        ddof=1
    )


    pooled_scale = np.sqrt(
        0.5
        *
        (
            nominal_std**2
            +
            fault_std**2
        )
    )


    if pooled_scale > 0.0:

        standardized_separation = (
            abs(
                fault_mean
                -
                nominal_mean
            )
            /
            pooled_scale
        )

    else:

        standardized_separation = (
            0.0
        )


    feature_summary_rows.append(
        {
            "feature":
                feature_name,

            "nominal_mean":
                nominal_mean,

            "nominal_std":
                nominal_std,

            "fault_mean":
                fault_mean,

            "fault_std":
                fault_std,

            "standardized_separation":
                standardized_separation
        }
    )


feature_summary = pd.DataFrame(
    feature_summary_rows
)


feature_summary = (
    feature_summary.sort_values(
        by=
            "standardized_separation",

        ascending=
            False
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 16. SAVE DATA
# ------------------------------------------------------------

dataset.to_csv(
    dataset_path,
    index=False
)


summary.to_csv(
    summary_path,
    index=False
)


feature_summary.to_csv(
    feature_summary_path,
    index=False
)


# ------------------------------------------------------------
# 17. DIAGNOSTIC DETECTION RATES
# ------------------------------------------------------------

five_sat_summary = (
    summary[
        summary[
            "redundancy_mode"
        ]
        ==
        "FIVE_SAT"
    ]
)


full_summary = (
    summary[
        summary[
            "redundancy_mode"
        ]
        ==
        "FULL"
    ]
)


# ------------------------------------------------------------
# 18. VALIDATION
# ------------------------------------------------------------

dataset_size_valid = (
    len(
        dataset
    )
    >=
    1000
)


dataset_validation = (
    dataset_size_valid
    and
    all_features_finite
    and
    labels_not_in_features
    and
    all_groups_complete
    and
    both_classes_present
)


# ------------------------------------------------------------
# 19. PRINT RESULTS
# ------------------------------------------------------------

print()

print(
    "================================================================================================================"
)

print(
    "AURORA — Experience 015-A RESULTS"
)

print(
    "================================================================================================================"
)

print(
    f"Groups attempted : "
    f"{number_of_attempted_groups}"
)

print(
    f"Groups failed : "
    f"{number_of_failed_groups}"
)

print(
    f"Complete paired groups : "
    f"{number_of_complete_groups}"
)

print(
    f"Samples / group : "
    f"{expected_samples_per_group}"
)

print(
    f"Total samples : "
    f"{len(dataset)}"
)

print(
    f"Nominal samples : "
    f"{number_nominal_samples}"
)

print(
    f"Fault samples : "
    f"{number_fault_samples}"
)

print(
    f"Feature count : "
    f"{len(AI_FEATURE_COLUMNS)}"
)

print(
    f"All features finite : "
    f"{all_features_finite}"
)

print(
    f"Labels excluded from AI features : "
    f"{labels_not_in_features}"
)

print(
    f"All paired groups complete : "
    f"{all_groups_complete}"
)

print()


print(
    "----- CLASSICAL FDIR BASELINE -----"
)

print(
    summary.to_string(
        index=False,
        formatters={
            "fdir_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "fdir_acceptance_rate":
                lambda value:
                    f"{value:7.2f}",

            "mean_satellites":
                lambda value:
                    f"{value:7.2f}",

            "mean_pdop":
                lambda value:
                    f"{value:7.3f}",

            "mean_chi2_ratio":
                lambda value:
                    f"{value:8.3f}",

            "mean_residual_rms_m":
                lambda value:
                    f"{value:8.3f}",

            "mean_residual_max_abs_m":
                lambda value:
                    f"{value:8.3f}"
        }
    )
)

print()


print(
    "----- TOP FEATURE SEPARATIONS -----"
)

print(
    feature_summary[
        [
            "feature",
            "nominal_mean",
            "fault_mean",
            "standardized_separation"
        ]
    ]
    .head(
        10
    )
    .to_string(
        index=False,
        formatters={
            "nominal_mean":
                lambda value:
                    f"{value:.4f}",

            "fault_mean":
                lambda value:
                    f"{value:.4f}",

            "standardized_separation":
                lambda value:
                    f"{value:.3f}"
        }
    )
)

print()


print(
    "----- FILES -----"
)

print(
    f"Dataset : "
    f"{dataset_path}"
)

print(
    f"Summary : "
    f"{summary_path}"
)

print(
    f"Feature summary : "
    f"{feature_summary_path}"
)

print()


print(
    "----- VALIDATION -----"
)

print(
    f"Dataset >= 1000 samples : "
    f"{dataset_size_valid}"
)

print(
    f"Features finite : "
    f"{all_features_finite}"
)

print(
    f"No label leakage in AI feature list : "
    f"{labels_not_in_features}"
)

print(
    f"Paired groups complete : "
    f"{all_groups_complete}"
)

print(
    f"Both classes present : "
    f"{both_classes_present}"
)

print()

print(
    f"VALIDATION GLOBALE 015-A : "
    f"{dataset_validation}"
)

print(
    "================================================================================================================"
)


# ------------------------------------------------------------
# 20. FIGURE — FDIR DETECTION VS FAULT MAGNITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    full_summary[
        "fault_magnitude_m"
    ],

    full_summary[
        "fdir_detection_rate"
    ],

    marker="o",
    label="FULL redundancy"
)


plt.plot(
    five_sat_summary[
        "fault_magnitude_m"
    ],

    five_sat_summary[
        "fdir_detection_rate"
    ],

    marker="o",
    label="5 satellites"
)


plt.xlabel(
    "Amplitude faute pseudorange [m]"
)

plt.ylabel(
    "Detection FDIR [%]"
)

plt.title(
    "AURORA — Classical FDIR detection baseline"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


detection_figure_path = (
    figure_directory
    /
    "phase15_015a_fdir_detection_baseline.png"
)


plt.savefig(
    detection_figure_path,
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 21. FIGURE — CHI2 FEATURE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


for redundancy_mode in [
    "FULL",
    "FIVE_SAT"
]:

    subset = (
        dataset[
            dataset[
                "redundancy_mode"
            ]
            ==
            redundancy_mode
        ]
    )


    grouped = (
        subset.groupby(
            "fault_magnitude_m"
        )[
            "chi2_ratio"
        ]
        .mean()
    )


    plt.plot(
        grouped.index,
        grouped.values,
        marker="o",
        label=
            redundancy_mode
    )


plt.axhline(
    1.0,
    linestyle="--",
    label="Classical chi2 threshold"
)


plt.xlabel(
    "Amplitude faute pseudorange [m]"
)

plt.ylabel(
    "Mean chi2 statistic / threshold"
)

plt.title(
    "AURORA — Chi-square anomaly feature"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


chi2_figure_path = (
    figure_directory
    /
    "phase15_015a_chi2_feature.png"
)


plt.savefig(
    chi2_figure_path,
    dpi=200
)

plt.show()