from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

from src.ai.isolation_forest_detector import (
    IsolationForestAnomalyDetector
)


# ============================================================
# AURORA
# Experience 015-B
#
# CLASSICAL FDIR VS ISOLATION FOREST
#
#
# Methodology:
#
#   TRAIN
#       runs 0, 1, 2
#       NOMINAL DATA ONLY
#
#   CALIBRATION
#       run 3
#       NOMINAL DATA ONLY
#
#   TEST
#       runs 4, 5
#       nominal + faults
#
#
# Therefore:
#
#   - no test leakage
#   - no fault labels used for training
#   - no fault labels used for threshold calibration
#
#
# Target AI false-alarm rate:
#
#       1 %
#
# approximately matching the classical 99% chi-square
# confidence level.
#
#
# NOTE:
#
# This is an INSTANTANEOUS ML baseline.
#
# Temporal features come later in 015-C.
# ============================================================


# ------------------------------------------------------------
# 1. FILES
# ------------------------------------------------------------

dataset_path = (
    Path(
        "data"
    )
    /
    "phase15"
    /
    "gnss_anomaly_dataset_015a.csv"
)

figure_directory = (
    Path(
        "results"
    )
    /
    "figures"
)

table_directory = (
    Path(
        "results"
    )
    /
    "tables"
)

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)

table_directory.mkdir(
    parents=True,
    exist_ok=True
)


if not dataset_path.exists():

    raise FileNotFoundError(
        f"Dataset introuvable : {dataset_path}"
    )


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

dataset = pd.read_csv(
    dataset_path
)


# ------------------------------------------------------------
# 3. COMPACT ML FEATURE SET
#
# 015-A showed several exact / near duplicates:
#
#   residual_rms_m
#   residual_std_m
#   normalized_residual_rms
#
# and:
#
#   number_of_satellites
#   degrees_of_freedom
#
# We deliberately reduce redundancy before ML.
# ------------------------------------------------------------

ML_FEATURE_COLUMNS = [
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
# 4. SPLIT BY ENTIRE NOISE RUN
#
# This is stricter than random row splitting.
#
# Same paired group never appears in more than one subset.
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
    5
]


training_dataset = dataset[
    dataset[
        "run_id"
    ].isin(
        training_run_ids
    )
].copy()


calibration_dataset = dataset[
    dataset[
        "run_id"
    ].isin(
        calibration_run_ids
    )
].copy()


test_dataset = dataset[
    dataset[
        "run_id"
    ].isin(
        test_run_ids
    )
].copy()


# ------------------------------------------------------------
# 5. NOMINAL-ONLY TRAIN / CALIBRATION
# ------------------------------------------------------------

training_nominal = training_dataset[
    training_dataset[
        "fault_label"
    ]
    ==
    0
].copy()


calibration_nominal = calibration_dataset[
    calibration_dataset[
        "fault_label"
    ]
    ==
    0
].copy()


# ------------------------------------------------------------
# 6. SPLIT VALIDATION
# ------------------------------------------------------------

train_group_ids = set(
    training_dataset[
        "group_id"
    ]
)

calibration_group_ids = set(
    calibration_dataset[
        "group_id"
    ]
)

test_group_ids = set(
    test_dataset[
        "group_id"
    ]
)


train_calibration_overlap = len(
    train_group_ids.intersection(
        calibration_group_ids
    )
)


train_test_overlap = len(
    train_group_ids.intersection(
        test_group_ids
    )
)


calibration_test_overlap = len(
    calibration_group_ids.intersection(
        test_group_ids
    )
)


splits_disjoint = (
    train_calibration_overlap
    ==
    0
    and
    train_test_overlap
    ==
    0
    and
    calibration_test_overlap
    ==
    0
)


training_is_nominal_only = bool(
    np.all(
        training_nominal[
            "fault_label"
        ].to_numpy()
        ==
        0
    )
)


calibration_is_nominal_only = bool(
    np.all(
        calibration_nominal[
            "fault_label"
        ].to_numpy()
        ==
        0
    )
)


# ------------------------------------------------------------
# 7. MODEL
# ------------------------------------------------------------

detector = (
    IsolationForestAnomalyDetector(
        feature_columns=
            ML_FEATURE_COLUMNS,

        n_estimators=
            400,

        random_state=
            1502
    )
)


detector.fit_nominal(
    training_nominal
)


# ------------------------------------------------------------
# 8. THRESHOLD CALIBRATION
# ------------------------------------------------------------

target_false_alarm_rate = (
    0.01
)


calibration_result = (
    detector.calibrate_threshold(
        nominal_calibration_dataframe=
            calibration_nominal,

        target_false_alarm_rate=
            target_false_alarm_rate
    )
)


# ------------------------------------------------------------
# 9. TEST SCORES
# ------------------------------------------------------------

test_scores = (
    detector.score_samples(
        test_dataset
    )
)


test_ai_predictions = (
    detector.predict(
        test_dataset
    )
)


test_labels = (
    test_dataset[
        "fault_label"
    ].to_numpy(
        dtype=int
    )
)


test_fdir_predictions = (
    test_dataset[
        "fdir_detected"
    ].to_numpy(
        dtype=int
    )
)


test_dataset[
    "ai_anomaly_score"
] = (
    test_scores
)


test_dataset[
    "ai_detected"
] = (
    test_ai_predictions
)


# ------------------------------------------------------------
# 10. GLOBAL BINARY METRICS
# ------------------------------------------------------------

def binary_metrics(
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


    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    return {
        "false_alarm_rate":
            false_alarm_rate,

        "detection_rate":
            detection_rate,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    }


fdir_metrics = (
    binary_metrics(
        labels=
            test_labels,

        predictions=
            test_fdir_predictions
    )
)


ai_metrics = (
    binary_metrics(
        labels=
            test_labels,

        predictions=
            test_ai_predictions
    )
)


# ------------------------------------------------------------
# 11. CONTINUOUS AI SCORE METRICS
# ------------------------------------------------------------

ai_roc_auc = roc_auc_score(
    test_labels,
    test_scores
)


ai_average_precision = average_precision_score(
    test_labels,
    test_scores
)


# ------------------------------------------------------------
# 12. PERFORMANCE BY REDUNDANCY / FAULT MAGNITUDE
# ------------------------------------------------------------

comparison_rows = []


for redundancy_mode in [
    "FULL",
    "FIVE_SAT"
]:

    for fault_magnitude in sorted(
        test_dataset[
            "fault_magnitude_m"
        ].unique()
    ):

        subset = test_dataset[
            (
                test_dataset[
                    "redundancy_mode"
                ]
                ==
                redundancy_mode
            )
            &
            (
                test_dataset[
                    "fault_magnitude_m"
                ]
                ==
                fault_magnitude
            )
        ]


        if len(
            subset
        ) == 0:

            continue


        fdir_detection_rate = (
            100.0
            *
            np.mean(
                subset[
                    "fdir_detected"
                ]
            )
        )


        ai_detection_rate = (
            100.0
            *
            np.mean(
                subset[
                    "ai_detected"
                ]
            )
        )


        mean_ai_score = np.mean(
            subset[
                "ai_anomaly_score"
            ]
        )


        comparison_rows.append(
            {
                "redundancy_mode":
                    redundancy_mode,

                "fault_magnitude_m":
                    fault_magnitude,

                "samples":
                    len(
                        subset
                    ),

                "fdir_detection_rate":
                    fdir_detection_rate,

                "ai_detection_rate":
                    ai_detection_rate,

                "ai_minus_fdir":
                    (
                        ai_detection_rate
                        -
                        fdir_detection_rate
                    ),

                "mean_ai_score":
                    mean_ai_score
            }
        )


comparison_table = pd.DataFrame(
    comparison_rows
)


# ------------------------------------------------------------
# 13. DIFFICULT SUBSETS
# ------------------------------------------------------------

five_sat_fault_mask = (
    (
        test_dataset[
            "redundancy_mode"
        ]
        ==
        "FIVE_SAT"
    )
    &
    (
        test_dataset[
            "fault_label"
        ]
        ==
        1
    )
)


five_sat_50m_mask = (
    (
        test_dataset[
            "redundancy_mode"
        ]
        ==
        "FIVE_SAT"
    )
    &
    (
        test_dataset[
            "fault_magnitude_m"
        ]
        ==
        50.0
    )
)


five_sat_20m_mask = (
    (
        test_dataset[
            "redundancy_mode"
        ]
        ==
        "FIVE_SAT"
    )
    &
    (
        test_dataset[
            "fault_magnitude_m"
        ]
        ==
        20.0
    )
)


full_20m_mask = (
    (
        test_dataset[
            "redundancy_mode"
        ]
        ==
        "FULL"
    )
    &
    (
        test_dataset[
            "fault_magnitude_m"
        ]
        ==
        20.0
    )
)


def subset_detection_rates(
    mask
):

    subset = test_dataset[
        mask
    ]


    if len(
        subset
    ) == 0:

        return (
            np.nan,
            np.nan
        )


    return (
        100.0
        *
        np.mean(
            subset[
                "fdir_detected"
            ]
        ),

        100.0
        *
        np.mean(
            subset[
                "ai_detected"
            ]
        )
    )


(
    five_sat_fault_fdir,
    five_sat_fault_ai
) = subset_detection_rates(
    five_sat_fault_mask
)


(
    five_sat_50_fdir,
    five_sat_50_ai
) = subset_detection_rates(
    five_sat_50m_mask
)


(
    five_sat_20_fdir,
    five_sat_20_ai
) = subset_detection_rates(
    five_sat_20m_mask
)


(
    full_20_fdir,
    full_20_ai
) = subset_detection_rates(
    full_20m_mask
)


# ------------------------------------------------------------
# 14. NOMINAL FALSE ALARMS BY REDUNDANCY
# ------------------------------------------------------------

nominal_test = test_dataset[
    test_dataset[
        "fault_label"
    ]
    ==
    0
]


nominal_false_alarm_rows = []


for redundancy_mode in [
    "FULL",
    "FIVE_SAT"
]:

    subset = nominal_test[
        nominal_test[
            "redundancy_mode"
        ]
        ==
        redundancy_mode
    ]


    nominal_false_alarm_rows.append(
        {
            "redundancy_mode":
                redundancy_mode,

            "samples":
                len(
                    subset
                ),

            "fdir_false_alarm_rate":
                100.0
                *
                np.mean(
                    subset[
                        "fdir_detected"
                    ]
                ),

            "ai_false_alarm_rate":
                100.0
                *
                np.mean(
                    subset[
                        "ai_detected"
                    ]
                )
        }
    )


nominal_false_alarm_table = pd.DataFrame(
    nominal_false_alarm_rows
)


# ------------------------------------------------------------
# 15. SCORE DISTRIBUTIONS
# ------------------------------------------------------------

nominal_scores = (
    test_dataset.loc[
        test_dataset[
            "fault_label"
        ]
        ==
        0,
        "ai_anomaly_score"
    ].to_numpy()
)


fault_scores = (
    test_dataset.loc[
        test_dataset[
            "fault_label"
        ]
        ==
        1,
        "ai_anomaly_score"
    ].to_numpy()
)


mean_nominal_score = np.mean(
    nominal_scores
)


mean_fault_score = np.mean(
    fault_scores
)


# ------------------------------------------------------------
# 16. SAVE TABLES
# ------------------------------------------------------------

comparison_path = (
    table_directory
    /
    "phase15_015b_fdir_vs_isolation_forest.csv"
)


false_alarm_path = (
    table_directory
    /
    "phase15_015b_false_alarm_by_redundancy.csv"
)


test_predictions_path = (
    table_directory
    /
    "phase15_015b_test_predictions.csv"
)


comparison_table.to_csv(
    comparison_path,
    index=False
)


nominal_false_alarm_table.to_csv(
    false_alarm_path,
    index=False
)


test_dataset.to_csv(
    test_predictions_path,
    index=False
)


# ------------------------------------------------------------
# 17. METHODOLOGICAL VALIDATION
# ------------------------------------------------------------

test_false_alarm_reasonable = (
    ai_metrics[
        "false_alarm_rate"
    ]
    <=
    5.0
)


score_has_signal = (
    ai_roc_auc
    >
    0.60
)


fault_scores_higher = (
    mean_fault_score
    >
    mean_nominal_score
)


methodology_valid = (
    splits_disjoint
    and
    training_is_nominal_only
    and
    calibration_is_nominal_only
    and
    test_false_alarm_reasonable
    and
    score_has_signal
    and
    fault_scores_higher
)


# ------------------------------------------------------------
# 18. DOES AI ADD DETECTION VALUE?
#
# This is NOT required for methodology validation.
#
# It is a scientific result of the experiment.
# ------------------------------------------------------------

ai_improves_five_sat_50m = (
    five_sat_50_ai
    >
    five_sat_50_fdir
)


ai_improves_five_sat_overall = (
    five_sat_fault_ai
    >
    five_sat_fault_fdir
)


# ------------------------------------------------------------
# 19. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)

print(
    "AURORA — Experience 015-B"
)

print(
    "Classical FDIR vs Isolation Forest — instantaneous anomaly detection"
)

print(
    "========================================================================================================================"
)

print(
    f"Dataset : "
    f"{dataset_path}"
)

print(
    f"Features ML : "
    f"{len(ML_FEATURE_COLUMNS)}"
)

print()

print(
    "----- SPLIT -----"
)

print(
    f"Train runs : "
    f"{training_run_ids}"
)

print(
    f"Calibration runs : "
    f"{calibration_run_ids}"
)

print(
    f"Test runs : "
    f"{test_run_ids}"
)

print(
    f"Train rows : "
    f"{len(training_dataset)}"
)

print(
    f"Train NOMINAL rows actually used : "
    f"{len(training_nominal)}"
)

print(
    f"Calibration nominal rows : "
    f"{len(calibration_nominal)}"
)

print(
    f"Test rows : "
    f"{len(test_dataset)}"
)

print(
    f"Group overlap train/cal/test : "
    f"{train_calibration_overlap}/"
    f"{train_test_overlap}/"
    f"{calibration_test_overlap}"
)

print()


print(
    "----- THRESHOLD CALIBRATION -----"
)

print(
    f"Target false alarm : "
    f"{100.0 * target_false_alarm_rate:.2f} %"
)

print(
    f"Threshold anomaly score : "
    f"{calibration_result['threshold']:.6f}"
)

print(
    f"Achieved calibration false alarm : "
    f"{100.0 * calibration_result['achieved_false_alarm_rate']:.2f} %"
)

print()


print(
    "----- GLOBAL TEST PERFORMANCE -----"
)

header = (
    f"{'Detector':>20} | "
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

print(
    f"{'Classical FDIR':>20} | "
    f"{fdir_metrics['false_alarm_rate']:10.2f} | "
    f"{fdir_metrics['detection_rate']:10.2f} | "
    f"{fdir_metrics['precision']:10.3f} | "
    f"{fdir_metrics['recall']:10.3f} | "
    f"{fdir_metrics['f1']:10.3f}"
)

print(
    f"{'Isolation Forest':>20} | "
    f"{ai_metrics['false_alarm_rate']:10.2f} | "
    f"{ai_metrics['detection_rate']:10.2f} | "
    f"{ai_metrics['precision']:10.3f} | "
    f"{ai_metrics['recall']:10.3f} | "
    f"{ai_metrics['f1']:10.3f}"
)

print()

print(
    f"AI ROC-AUC : "
    f"{ai_roc_auc:.4f}"
)

print(
    f"AI Average Precision : "
    f"{ai_average_precision:.4f}"
)

print(
    f"Mean anomaly score nominal : "
    f"{mean_nominal_score:.6f}"
)

print(
    f"Mean anomaly score fault : "
    f"{mean_fault_score:.6f}"
)

print()


print(
    "----- PERFORMANCE PAR REGIME / AMPLITUDE -----"
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

            "ai_detection_rate":
                lambda value:
                    f"{value:7.2f}",

            "ai_minus_fdir":
                lambda value:
                    f"{value:+7.2f}",

            "mean_ai_score":
                lambda value:
                    f"{value:.6f}"
        }
    )
)

print()


print(
    "----- NOMINAL FALSE ALARMS -----"
)

print(
    nominal_false_alarm_table.to_string(
        index=False,
        formatters={
            "fdir_false_alarm_rate":
                lambda value:
                    f"{value:7.2f}",

            "ai_false_alarm_rate":
                lambda value:
                    f"{value:7.2f}"
        }
    )
)

print()


print(
    "----- DIFFICULT SUBSETS -----"
)

print(
    f"FIVE_SAT all faults "
    f"FDIR / AI : "
    f"{five_sat_fault_fdir:.2f} / "
    f"{five_sat_fault_ai:.2f} %"
)

print(
    f"FIVE_SAT +20 m "
    f"FDIR / AI : "
    f"{five_sat_20_fdir:.2f} / "
    f"{five_sat_20_ai:.2f} %"
)

print(
    f"FIVE_SAT +50 m "
    f"FDIR / AI : "
    f"{five_sat_50_fdir:.2f} / "
    f"{five_sat_50_ai:.2f} %"
)

print(
    f"FULL +20 m "
    f"FDIR / AI : "
    f"{full_20_fdir:.2f} / "
    f"{full_20_ai:.2f} %"
)

print()


print(
    "----- SCIENTIFIC INTERPRETATION FLAGS -----"
)

print(
    f"AI improves FIVE_SAT overall : "
    f"{ai_improves_five_sat_overall}"
)

print(
    f"AI improves FIVE_SAT +50 m : "
    f"{ai_improves_five_sat_50m}"
)

print()


print(
    "----- VALIDATION METHODOLOGIQUE -----"
)

print(
    f"Splits disjoint : "
    f"{splits_disjoint}"
)

print(
    f"Training nominal only : "
    f"{training_is_nominal_only}"
)

print(
    f"Calibration nominal only : "
    f"{calibration_is_nominal_only}"
)

print(
    f"Test false alarm <= 5% : "
    f"{test_false_alarm_reasonable}"
)

print(
    f"ROC-AUC > 0.60 : "
    f"{score_has_signal}"
)

print(
    f"Fault score > nominal score : "
    f"{fault_scores_higher}"
)

print()

print(
    f"VALIDATION GLOBALE 015-B : "
    f"{methodology_valid}"
)

print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 20. ROC CURVE
# ------------------------------------------------------------

false_positive_rate, true_positive_rate, _ = (
    roc_curve(
        test_labels,
        test_scores
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=
        f"Isolation Forest AUC = {ai_roc_auc:.3f}"
)


plt.plot(
    [
        0.0,
        1.0
    ],
    [
        0.0,
        1.0
    ],
    linestyle="--",
    label="Random"
)


plt.xlabel(
    "False positive rate"
)

plt.ylabel(
    "True positive rate"
)

plt.title(
    "AURORA — 015-B Isolation Forest ROC"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


roc_path = (
    figure_directory
    /
    "phase15_015b_roc_curve.png"
)


plt.savefig(
    roc_path,
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 21. PRECISION-RECALL CURVE
# ------------------------------------------------------------

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        test_labels,
        test_scores
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    recall_curve,
    precision_curve,
    label=
        f"AP = {ai_average_precision:.3f}"
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "AURORA — 015-B Precision-Recall"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


pr_path = (
    figure_directory
    /
    "phase15_015b_precision_recall.png"
)


plt.savefig(
    pr_path,
    dpi=200
)

plt.show()


# ------------------------------------------------------------
# 22. DETECTION VS FAULT MAGNITUDE
# ------------------------------------------------------------

for redundancy_mode in [
    "FULL",
    "FIVE_SAT"
]:

    subset = comparison_table[
        comparison_table[
            "redundancy_mode"
        ]
        ==
        redundancy_mode
    ]


    plt.figure(
        figsize=(9, 6)
    )


    plt.plot(
        subset[
            "fault_magnitude_m"
        ],

        subset[
            "fdir_detection_rate"
        ],

        marker="o",
        label="Classical FDIR"
    )


    plt.plot(
        subset[
            "fault_magnitude_m"
        ],

        subset[
            "ai_detection_rate"
        ],

        marker="o",
        label="Isolation Forest"
    )


    plt.xlabel(
        "Amplitude faute pseudorange [m]"
    )

    plt.ylabel(
        "Detection [%]"
    )

    plt.title(
        f"AURORA — 015-B {redundancy_mode}"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()


    figure_path = (
        figure_directory
        /
        f"phase15_015b_detection_{redundancy_mode.lower()}.png"
    )


    plt.savefig(
        figure_path,
        dpi=200
    )

    plt.show()


# ------------------------------------------------------------
# 23. SCORE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    nominal_scores,
    bins=30,
    alpha=0.6,
    label="Nominal"
)


plt.hist(
    fault_scores,
    bins=30,
    alpha=0.6,
    label="Fault"
)


plt.axvline(
    detector.threshold,
    linestyle="--",
    label="Calibrated threshold"
)


plt.xlabel(
    "Isolation Forest anomaly score"
)

plt.ylabel(
    "Samples"
)

plt.title(
    "AURORA — 015-B anomaly-score distributions"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


score_path = (
    figure_directory
    /
    "phase15_015b_score_distribution.png"
)


plt.savefig(
    score_path,
    dpi=200
)

plt.show()