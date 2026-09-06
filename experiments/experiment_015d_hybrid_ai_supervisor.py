from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from src.ai.isolation_forest_detector import (
    IsolationForestAnomalyDetector
)

from src.ai.gnss_temporal_features import (
    TEMPORAL_FEATURE_COLUMNS
)

from src.ai.hybrid_anomaly_supervisor import (
    HybridGnssAnomalySupervisor
)


# ============================================================
# AURORA
# Experiment 015-D V2
#
# PERSISTENCE-AWARE FDIR + AI SUPERVISOR
#
#
# IMPORTANT SCIENTIFIC CORRECTION
#
# A 2-out-of-3 temporal voting rule may maintain a hard alarm
# for one epoch after the instantaneous AI alarm disappears.
#
# Therefore:
#
#   P_FA(epoch, persistence)
#
# is NOT mathematically guaranteed to be lower than:
#
#   P_FA(epoch, naive OR)
#
#
# The correct operational questions are:
#
#   1. Is absolute P_FA still acceptable?
#   2. Does detection improve?
#   3. Does temporal voting avoid increasing the number
#      of nuisance-alarm episodes?
#
#
# This experiment validates those quantities.
# ============================================================


# ------------------------------------------------------------
# 1. FILES
# ------------------------------------------------------------

dataset_path = (
    Path("data")
    /
    "phase15"
    /
    "gnss_temporal_dataset_015c.csv"
)

table_directory = (
    Path("results")
    /
    "tables"
)

table_directory.mkdir(
    parents=True,
    exist_ok=True
)

prediction_path = (
    table_directory
    /
    "phase15_015d_hybrid_predictions.csv"
)

comparison_path = (
    table_directory
    /
    "phase15_015d_hybrid_detection.csv"
)


if not dataset_path.exists():

    raise FileNotFoundError(
        f"Dataset introuvable : {dataset_path}"
    )


# ------------------------------------------------------------
# 2. LOAD
# ------------------------------------------------------------

dataset = pd.read_csv(
    dataset_path
)

dataset = dataset[
    dataset[
        "temporal_features_valid"
    ]
    ==
    1
].copy()


# ------------------------------------------------------------
# 3. DATA SPLIT
# ------------------------------------------------------------

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


training_nominal = dataset[
    (
        dataset[
            "run_id"
        ].isin(
            training_run_ids
        )
    )
    &
    (
        dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


calibration_nominal = dataset[
    (
        dataset[
            "run_id"
        ].isin(
            calibration_run_ids
        )
    )
    &
    (
        dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


test_sequences = dataset[
    dataset[
        "run_id"
    ].isin(
        test_run_ids
    )
].copy()


# ------------------------------------------------------------
# 4. SPLIT VALIDATION
# ------------------------------------------------------------

training_runs = set(
    training_nominal[
        "run_id"
    ].unique()
)

calibration_runs = set(
    calibration_nominal[
        "run_id"
    ].unique()
)

test_runs = set(
    test_sequences[
        "run_id"
    ].unique()
)


splits_disjoint = (
    len(
        training_runs
        &
        calibration_runs
    )
    ==
    0
    and
    len(
        training_runs
        &
        test_runs
    )
    ==
    0
    and
    len(
        calibration_runs
        &
        test_runs
    )
    ==
    0
)


training_nominal_only = bool(
    np.all(
        training_nominal[
            "fault_active"
        ].to_numpy()
        ==
        0
    )
)


calibration_nominal_only = bool(
    np.all(
        calibration_nominal[
            "fault_active"
        ].to_numpy()
        ==
        0
    )
)


# ------------------------------------------------------------
# 5. TRAIN TEMPORAL ISOLATION FOREST
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
# 6. SORT TEST SEQUENCES
# ------------------------------------------------------------

test_sequences = (
    test_sequences.sort_values(
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


# ------------------------------------------------------------
# 7. RAW TEMPORAL AI
# ------------------------------------------------------------

test_sequences[
    "temporal_score"
] = (
    detector.score_samples(
        test_sequences
    )
)


test_sequences[
    "ai_raw_alarm"
] = (
    test_sequences[
        "temporal_score"
    ]
    >=
    detector.threshold
).astype(
    int
)


# ------------------------------------------------------------
# 8. SUPERVISOR SETTINGS
# ------------------------------------------------------------

ai_vote_window = (
    3
)

minimum_ai_votes = (
    2
)

dt = (
    10.0
)


# ------------------------------------------------------------
# 9. RUN HYBRID SUPERVISOR
# ------------------------------------------------------------

number_of_rows = len(
    test_sequences
)


naive_or_alarm = np.zeros(
    number_of_rows,
    dtype=int
)

persistence_hard_alarm = np.zeros(
    number_of_rows,
    dtype=int
)

ai_confirmed = np.zeros(
    number_of_rows,
    dtype=int
)

ai_vote_count = np.zeros(
    number_of_rows,
    dtype=int
)

supervisor_state = np.empty(
    number_of_rows,
    dtype=object
)


grouped_indices = (
    test_sequences.groupby(
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
                test_sequences.loc[
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
            test_sequences.loc[
                index,
                "fdir_detected"
            ]
        )


        raw_ai_alarm = bool(
            test_sequences.loc[
                index,
                "ai_raw_alarm"
            ]
        )


        decision = (
            supervisor.update(
                classical_alarm=
                    classical_alarm,

                ai_raw_alarm=
                    raw_ai_alarm
            )
        )


        naive_or_alarm[
            index
        ] = int(
            classical_alarm
            or
            raw_ai_alarm
        )


        persistence_hard_alarm[
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


        supervisor_state[
            index
        ] = (
            decision.state.value
        )


test_sequences[
    "naive_or_alarm"
] = (
    naive_or_alarm
)


test_sequences[
    "persistence_hard_alarm"
] = (
    persistence_hard_alarm
)


test_sequences[
    "ai_confirmed"
] = (
    ai_confirmed
)


test_sequences[
    "ai_vote_count"
] = (
    ai_vote_count
)


test_sequences[
    "supervisor_state"
] = (
    supervisor_state
)


# ------------------------------------------------------------
# 10. FAIR EVALUATION DATASET
#
# Nominal:
#       only fault_magnitude = 0 sequence
#
# Fault:
#       only epochs where an injected fault is active
# ------------------------------------------------------------

nominal_evaluation = test_sequences[
    test_sequences[
        "fault_magnitude_m"
    ]
    ==
    0.0
].copy()


fault_evaluation = test_sequences[
    test_sequences[
        "fault_active"
    ]
    ==
    1
].copy()


evaluation_dataset = pd.concat(
    (
        nominal_evaluation,
        fault_evaluation
    ),
    ignore_index=True
)


# ------------------------------------------------------------
# 11. BINARY METRICS
# ------------------------------------------------------------

def compute_binary_metrics(
    dataframe,
    prediction_column
):

    labels = (
        dataframe[
            "fault_active"
        ].to_numpy(
            dtype=int
        )
    )


    predictions = (
        dataframe[
            prediction_column
        ].to_numpy(
            dtype=int
        )
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


    false_alarm_rate = (
        100.0
        *
        np.mean(
            predictions[
                nominal_mask
            ]
        )
    )


    detection_rate = (
        100.0
        *
        np.mean(
            predictions[
                fault_mask
            ]
        )
    )


    return {
        "false_alarm_rate":
            false_alarm_rate,

        "detection_rate":
            detection_rate,

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
        evaluation_dataset,
        "fdir_detected"
    )
)


temporal_ai_metrics = (
    compute_binary_metrics(
        evaluation_dataset,
        "ai_raw_alarm"
    )
)


naive_or_metrics = (
    compute_binary_metrics(
        evaluation_dataset,
        "naive_or_alarm"
    )
)


persistence_metrics = (
    compute_binary_metrics(
        evaluation_dataset,
        "persistence_hard_alarm"
    )
)


# ------------------------------------------------------------
# 12. PERFORMANCE BY FAULT MAGNITUDE
# ------------------------------------------------------------

comparison_rows = []


for fault_magnitude_m in [
    5.0,
    10.0,
    20.0,
    50.0
]:

    subset = fault_evaluation[
        fault_evaluation[
            "fault_magnitude_m"
        ]
        ==
        fault_magnitude_m
    ]


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

            "temporal_ai_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "ai_raw_alarm"
                    ]
                ),

            "naive_or_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "naive_or_alarm"
                    ]
                ),

            "persistence_detection_rate":
                100.0
                *
                np.mean(
                    subset[
                        "persistence_hard_alarm"
                    ]
                )
        }
    )


comparison_table = pd.DataFrame(
    comparison_rows
)


# ------------------------------------------------------------
# 13. ALARM EPISODE COUNTER
# ------------------------------------------------------------

def count_alarm_episodes(
    dataframe,
    prediction_column
):

    total_episodes = (
        0
    )


    for run_id in test_run_ids:

        subset = (
            dataframe[
                dataframe[
                    "run_id"
                ]
                ==
                run_id
            ]
            .sort_values(
                "time_seconds"
            )
        )


        alarms = (
            subset[
                prediction_column
            ].to_numpy(
                dtype=int
            )
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

                total_episodes += (
                    1
                )


            previous_alarm = (
                alarm
            )


    return (
        total_episodes
    )


# ------------------------------------------------------------
# 14. ALARM-EPOCH COUNTER
# ------------------------------------------------------------

def count_alarm_epochs(
    dataframe,
    prediction_column
):

    return int(
        np.sum(
            dataframe[
                prediction_column
        ].to_numpy(
            dtype=int
        )
    )
)


# ------------------------------------------------------------
# 15. NOMINAL NUISANCE METRICS
# ------------------------------------------------------------

fdir_false_alarm_episodes = (
    count_alarm_episodes(
        nominal_evaluation,
        "fdir_detected"
    )
)


temporal_false_alarm_episodes = (
    count_alarm_episodes(
        nominal_evaluation,
        "ai_raw_alarm"
    )
)


naive_or_false_alarm_episodes = (
    count_alarm_episodes(
        nominal_evaluation,
        "naive_or_alarm"
    )
)


persistence_false_alarm_episodes = (
    count_alarm_episodes(
        nominal_evaluation,
        "persistence_hard_alarm"
    )
)


fdir_false_alarm_epochs = (
    count_alarm_epochs(
        nominal_evaluation,
        "fdir_detected"
    )
)


temporal_false_alarm_epochs = (
    count_alarm_epochs(
        nominal_evaluation,
        "ai_raw_alarm"
    )
)


naive_or_false_alarm_epochs = (
    count_alarm_epochs(
        nominal_evaluation,
        "naive_or_alarm"
    )
)


persistence_false_alarm_epochs = (
    count_alarm_epochs(
        nominal_evaluation,
        "persistence_hard_alarm"
    )
)


# ------------------------------------------------------------
# 16. EXPLAIN PERSISTENCE MEMORY
#
# Count hard alarms occurring when current raw AI = 0
# and classical FDIR = 0.
#
# These are alarm-memory epochs generated by the 2/3 vote.
# ------------------------------------------------------------

persistence_memory_mask = (
    (
        nominal_evaluation[
            "persistence_hard_alarm"
        ]
        ==
        1
    )
    &
    (
        nominal_evaluation[
            "ai_raw_alarm"
        ]
        ==
        0
    )
    &
    (
        nominal_evaluation[
            "fdir_detected"
        ]
        ==
        0
    )
)


nominal_persistence_memory_epochs = int(
    np.sum(
        persistence_memory_mask
    )
)


fault_persistence_memory_mask = (
    (
        fault_evaluation[
            "persistence_hard_alarm"
        ]
        ==
        1
    )
    &
    (
        fault_evaluation[
            "ai_raw_alarm"
        ]
        ==
        0
    )
    &
    (
        fault_evaluation[
            "fdir_detected"
        ]
        ==
        0
    )
)


fault_persistence_memory_epochs = int(
    np.sum(
        fault_persistence_memory_mask
    )
)


# ------------------------------------------------------------
# 17. +50 M PERFORMANCE
# ------------------------------------------------------------

fifty_meter_row = (
    comparison_table[
        comparison_table[
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
# 18. SCIENTIFIC FLAGS
# ------------------------------------------------------------

hybrid_improves_global_detection_vs_fdir = (
    persistence_metrics[
        "detection_rate"
    ]
    >
    fdir_metrics[
        "detection_rate"
    ]
)


hybrid_improves_global_detection_vs_naive = (
    persistence_metrics[
        "detection_rate"
    ]
    >
    naive_or_metrics[
        "detection_rate"
    ]
)


hybrid_improves_50m_vs_fdir = (
    fifty_meter_row[
        "persistence_detection_rate"
    ]
    >
    fifty_meter_row[
        "fdir_detection_rate"
    ]
)


nuisance_episode_count_not_increased = (
    persistence_false_alarm_episodes
    <=
    naive_or_false_alarm_episodes
)


absolute_false_alarm_rate_acceptable = (
    persistence_metrics[
        "false_alarm_rate"
    ]
    <=
    5.0
)


# ------------------------------------------------------------
# 19. VALIDATION
# ------------------------------------------------------------

methodology_valid = (
    splits_disjoint
    and
    training_nominal_only
    and
    calibration_nominal_only
)


validation_015d = (
    methodology_valid
    and
    absolute_false_alarm_rate_acceptable
    and
    hybrid_improves_global_detection_vs_fdir
    and
    hybrid_improves_50m_vs_fdir
    and
    nuisance_episode_count_not_increased
)


# ------------------------------------------------------------
# 20. SAVE
# ------------------------------------------------------------

test_sequences.to_csv(
    prediction_path,
    index=False
)


comparison_table.to_csv(
    comparison_path,
    index=False
)


# ------------------------------------------------------------
# 21. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-D V2"
)

print(
    "Persistence-aware FDIR + AI anomaly supervisor"
)

print(
    "========================================================================================================================"
)

print(
    f"AI vote rule : "
    f"{minimum_ai_votes}/{ai_vote_window}"
)

print(
    f"Temporal window : "
    f"{ai_vote_window * dt:.1f} s"
)

print(
    f"AI threshold : "
    f"{calibration['threshold']:.6f}"
)

print(
    f"Calibration P_FA : "
    f"{100.0 * calibration['achieved_false_alarm_rate']:.2f} %"
)

print(
    f"Splits disjoint : "
    f"{splits_disjoint}"
)

print()


print(
    "----- GLOBAL TEST PERFORMANCE -----"
)


header = (
    f"{'Detector':>24} | "
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


detector_rows = [
    (
        "Classical FDIR",
        fdir_metrics
    ),

    (
        "Temporal AI",
        temporal_ai_metrics
    ),

    (
        "Naive FDIR OR AI",
        naive_or_metrics
    ),

    (
        "Persistence hybrid",
        persistence_metrics
    )
]


for detector_name, metrics in detector_rows:

    print(
        f"{detector_name:>24} | "
        f"{metrics['false_alarm_rate']:10.2f} | "
        f"{metrics['detection_rate']:10.2f} | "
        f"{metrics['precision']:10.3f} | "
        f"{metrics['recall']:10.3f} | "
        f"{metrics['f1']:10.3f}"
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

            "temporal_ai_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "naive_or_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "persistence_detection_rate":
                lambda value:
                    f"{value:7.2f}"
        }
    )
)

print()


print(
    "----- NOMINAL FALSE-ALARM BEHAVIOUR -----"
)

print(
    f"Classical FDIR : "
    f"{fdir_false_alarm_epochs} alarm epochs, "
    f"{fdir_false_alarm_episodes} episodes"
)

print(
    f"Temporal AI : "
    f"{temporal_false_alarm_epochs} alarm epochs, "
    f"{temporal_false_alarm_episodes} episodes"
)

print(
    f"Naive OR : "
    f"{naive_or_false_alarm_epochs} alarm epochs, "
    f"{naive_or_false_alarm_episodes} episodes"
)

print(
    f"Persistence hybrid : "
    f"{persistence_false_alarm_epochs} alarm epochs, "
    f"{persistence_false_alarm_episodes} episodes"
)

print()


print(
    "----- TEMPORAL MEMORY -----"
)

print(
    f"Nominal hard-alarm epochs sustained only by vote memory : "
    f"{nominal_persistence_memory_epochs}"
)

print(
    f"Fault hard-alarm epochs sustained only by vote memory : "
    f"{fault_persistence_memory_epochs}"
)

print()

print(
    "Interpretation:"
)

print(
    "The 2/3 voting rule can extend an existing alarm "
    "without creating a new alarm episode."
)

print()


print(
    "----- SCIENTIFIC FLAGS -----"
)

print(
    f"Hybrid improves global detection vs FDIR : "
    f"{hybrid_improves_global_detection_vs_fdir}"
)

print(
    f"Hybrid improves global detection vs naive OR : "
    f"{hybrid_improves_global_detection_vs_naive}"
)

print(
    f"Hybrid improves +50 m vs FDIR : "
    f"{hybrid_improves_50m_vs_fdir}"
)

print(
    f"Nuisance episode count <= naive OR : "
    f"{nuisance_episode_count_not_increased}"
)

print(
    f"Absolute hybrid P_FA <= 5% : "
    f"{absolute_false_alarm_rate_acceptable}"
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
    f"Hybrid detection > classical FDIR : "
    f"{hybrid_improves_global_detection_vs_fdir}"
)

print(
    f"Hybrid +50 m detection > classical FDIR : "
    f"{hybrid_improves_50m_vs_fdir}"
)

print(
    f"Hybrid P_FA <= 5% : "
    f"{absolute_false_alarm_rate_acceptable}"
)

print(
    f"False-alarm episodes not increased vs naive OR : "
    f"{nuisance_episode_count_not_increased}"
)

print()

print(
    f"VALIDATION GLOBALE 015-D V2 : "
    f"{validation_015d}"
)

print(
    "========================================================================================================================"
)