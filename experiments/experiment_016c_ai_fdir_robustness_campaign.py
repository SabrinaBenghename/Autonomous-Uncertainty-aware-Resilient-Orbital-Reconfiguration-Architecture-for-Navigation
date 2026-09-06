from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    t as student_t
)

from src.ai.gnss_temporal_features import (
    TEMPORAL_FEATURE_COLUMNS
)

from src.ai.isolation_forest_detector import (
    IsolationForestAnomalyDetector
)

from src.validation.ai_robustness import (
    run_ai_robustness_scenario
)


# ============================================================
# AURORA
# Experience 016-C
#
# FROZEN AI / FDIR ROBUSTNESS CAMPAIGN
#
#
# Input:
#
#       64 validated Latin-Hypercube scenarios from 016-A
#
#
# Frozen detector:
#
#       Phase-15 temporal Isolation Forest
#
#       trained:
#           Phase-15 runs 0,1,2
#
#       calibrated:
#           Phase-15 run 3
#
#
# NO:
#
#       retraining
#       recalibration
#       threshold adaptation
#
#
# Main question:
#
# Does the Phase-15 AI benefit survive broader variation of:
#
#       GNSS pseudorange noise
#       fault magnitude
#       fault start
#       fault duration
#       small truth-trajectory variation
#
#
# The experiment is checkpointed after every scenario.
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

campaign_path = (
    Path("data")
    /
    "phase16"
    /
    "robustness_campaign_016a.csv"
)


phase15_dataset_path = (
    Path("data")
    /
    "phase15"
    /
    "gnss_temporal_dataset_015c.csv"
)


results_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016c_ai_fdir_robustness.csv"
)


summary_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016c_ai_summary.csv"
)


fault_quartile_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016c_fault_quartiles.csv"
)


noise_quartile_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016c_noise_quartiles.csv"
)


figure_directory = (
    Path("results")
    /
    "figures"
)


results_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


if not campaign_path.exists():

    raise FileNotFoundError(
        f"Missing 016-A campaign: {campaign_path}"
    )


if not phase15_dataset_path.exists():

    raise FileNotFoundError(
        f"Missing Phase-15 temporal dataset: "
        f"{phase15_dataset_path}"
    )


# ------------------------------------------------------------
# 2. LOAD FIXED CAMPAIGN
# ------------------------------------------------------------

campaign = pd.read_csv(
    campaign_path
)


number_of_scenarios = len(
    campaign
)


if number_of_scenarios != 64:

    raise RuntimeError(
        "016-C expects the validated 64-scenario "
        "016-A campaign."
    )


# ------------------------------------------------------------
# 3. REBUILD THE FROZEN PHASE-15 DETECTOR
#
# We intentionally reproduce EXACTLY the 015-C / 015-D
# training and threshold-calibration procedure.
# ------------------------------------------------------------

phase15_dataset = pd.read_csv(
    phase15_dataset_path
)


phase15_dataset = phase15_dataset[
    phase15_dataset[
        "temporal_features_valid"
    ]
    ==
    1
].copy()


training_run_ids = [
    0,
    1,
    2
]


calibration_run_ids = [
    3
]


training_nominal = phase15_dataset[
    (
        phase15_dataset[
            "run_id"
        ].isin(
            training_run_ids
        )
    )
    &
    (
        phase15_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


calibration_nominal = phase15_dataset[
    (
        phase15_dataset[
            "run_id"
        ].isin(
            calibration_run_ids
        )
    )
    &
    (
        phase15_dataset[
            "fault_magnitude_m"
        ]
        ==
        0.0
    )
].copy()


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


frozen_threshold = float(
    calibration[
        "threshold"
    ]
)


# ------------------------------------------------------------
# 4. CHECKPOINT
# ------------------------------------------------------------

if results_path.exists():

    existing_results = pd.read_csv(
        results_path
    )


    completed_ids = set(
        existing_results[
            "scenario_id"
        ].astype(
            str
        )
    )


else:

    existing_results = pd.DataFrame()

    completed_ids = set()


remaining_campaign = campaign[
    ~campaign[
        "scenario_id"
    ].astype(
        str
    ).isin(
        completed_ids
    )
].copy()


# ------------------------------------------------------------
# 5. HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 016-C"
)


print(
    "Frozen AI / FDIR robustness campaign"
)


print(
    "========================================================================================================================"
)


print(
    f"Scenarios : "
    f"{number_of_scenarios}"
)


print(
    f"Already completed : "
    f"{len(completed_ids)}"
)


print(
    f"Remaining : "
    f"{len(remaining_campaign)}"
)


print(
    f"Frozen temporal AI threshold : "
    f"{frozen_threshold:.6f}"
)


print(
    f"Original calibration P_FA : "
    f"{100.0 * calibration['achieved_false_alarm_rate']:.2f} %"
)


print()


# ------------------------------------------------------------
# 6. RUN REMAINING SCENARIOS
# ------------------------------------------------------------

new_rows = []


for local_index, (_, scenario) in enumerate(
    remaining_campaign.iterrows()
):

    scenario_id = str(
        scenario[
            "scenario_id"
        ]
    )


    global_index = (
        len(
            completed_ids
        )
        +
        local_index
        +
        1
    )


    print(
        f"[{global_index:02d}/{number_of_scenarios}] "
        f"{scenario_id} | "
        f"sigma_rho="
        f"{scenario['pseudorange_noise_std_m']:.2f} m | "
        f"fault="
        f"{scenario['fault_magnitude_m']:.1f} m | "
        f"start="
        f"{scenario['fault_start_min']:.1f} min | "
        f"duration="
        f"{scenario['fault_duration_min']:.1f} min"
    )


    result = (
        run_ai_robustness_scenario(
            scenario=
                scenario,

            detector=
                detector
        )
    )


    merged_result = {
        **scenario.to_dict(),
        **result,

        "frozen_ai_threshold":
            frozen_threshold
    }


    new_rows.append(
        merged_result
    )


    # ========================================================
    # Checkpoint after every scenario
    # ========================================================

    if len(
        existing_results
    ) > 0:

        checkpoint = pd.concat(
            (
                existing_results,
                pd.DataFrame(
                    new_rows
                )
            ),
            ignore_index=True
        )


    else:

        checkpoint = pd.DataFrame(
            new_rows
        )


    checkpoint = (
        checkpoint.sort_values(
            "scenario_index"
        )
        .drop_duplicates(
            subset=[
                "scenario_id"
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )


    checkpoint.to_csv(
        results_path,
        index=False
    )


    print(
        f"    P_D FDIR/AI/HYB = "
        f"{result['fdir_pd']:.1f} / "
        f"{result['temporal_ai_pd']:.1f} / "
        f"{result['hybrid_pd']:.1f} % | "
        f"P_FA AI/HYB = "
        f"{result['temporal_ai_pfa']:.1f} / "
        f"{result['hybrid_pfa']:.1f} %"
    )


# ------------------------------------------------------------
# 7. RELOAD COMPLETE RESULTS
# ------------------------------------------------------------

results = pd.read_csv(
    results_path
)


results = (
    results.sort_values(
        "scenario_index"
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 8. METHODOLOGICAL VALIDATION
# ------------------------------------------------------------

all_scenarios_complete = (
    len(
        results
    )
    ==
    number_of_scenarios
    and
    results[
        "scenario_id"
    ].nunique()
    ==
    number_of_scenarios
)


scenario_indices_complete = bool(
    np.array_equal(
        np.sort(
            results[
                "scenario_index"
            ].to_numpy(
                dtype=int
            )
        ),
        np.arange(
            number_of_scenarios,
            dtype=int
        )
    )
)


time_sync_valid = bool(
    np.max(
        np.abs(
            results[
                "time_sync_error_s"
            ].to_numpy(
                dtype=float
            )
        )
    )
    <
    1.0e-12
)


paired_sequences_complete = bool(
    np.all(
        results[
            "paired_sequences_complete"
        ].astype(
            bool
        )
    )
)


frozen_threshold_identical = bool(
    np.allclose(
        results[
            "frozen_ai_threshold"
        ].to_numpy(
            dtype=float
        ),
        frozen_threshold,
        rtol=0.0,
        atol=1.0e-12
    )
)


core_metric_columns = [
    "fdir_pfa",
    "temporal_ai_pfa",
    "hybrid_pfa",

    "fdir_pd",
    "temporal_ai_pd",
    "hybrid_pd",

    "hybrid_minus_fdir_pd",

    "nominal_mean_ai_score",
    "fault_mean_ai_score"
]


core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_metric_columns
            ].to_numpy(
                dtype=float
            )
        )
    )
)


# ------------------------------------------------------------
# 9. STATISTICAL HELPERS
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


    mean_value = float(
        np.mean(
            values
        )
    )


    if number < 2:

        return (
            mean_value,
            np.nan,
            np.nan,
            np.nan
        )


    std_value = float(
        np.std(
            values,
            ddof=1
        )
    )


    standard_error = (
        std_value
        /
        np.sqrt(
            number
        )
    )


    critical_value = (
        student_t.ppf(
            0.975,
            df=
                number
                -
                1
        )
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
# 10. ENSEMBLE PERFORMANCE
# ------------------------------------------------------------

(
    fdir_pd_mean,
    fdir_pd_std,
    fdir_pd_lower,
    fdir_pd_upper
) = mean_std_ci95(
    results[
        "fdir_pd"
    ]
)


(
    ai_pd_mean,
    ai_pd_std,
    ai_pd_lower,
    ai_pd_upper
) = mean_std_ci95(
    results[
        "temporal_ai_pd"
    ]
)


(
    hybrid_pd_mean,
    hybrid_pd_std,
    hybrid_pd_lower,
    hybrid_pd_upper
) = mean_std_ci95(
    results[
        "hybrid_pd"
    ]
)


paired_gain = (
    results[
        "hybrid_pd"
    ].to_numpy(
        dtype=float
    )
    -
    results[
        "fdir_pd"
    ].to_numpy(
        dtype=float
    )
)


(
    paired_gain_mean,
    paired_gain_std,
    paired_gain_lower,
    paired_gain_upper
) = mean_std_ci95(
    paired_gain
)


# ------------------------------------------------------------
# 11. FALSE ALARMS
# ------------------------------------------------------------

(
    fdir_pfa_mean,
    fdir_pfa_std,
    _,
    _
) = mean_std_ci95(
    results[
        "fdir_pfa"
    ]
)


(
    ai_pfa_mean,
    ai_pfa_std,
    ai_pfa_lower,
    ai_pfa_upper
) = mean_std_ci95(
    results[
        "temporal_ai_pfa"
    ]
)


(
    hybrid_pfa_mean,
    hybrid_pfa_std,
    hybrid_pfa_lower,
    hybrid_pfa_upper
) = mean_std_ci95(
    results[
        "hybrid_pfa"
    ]
)


# ------------------------------------------------------------
# 12. SCENARIO-BY-SCENARIO OUTCOMES
# ------------------------------------------------------------

hybrid_better_than_fdir = (
    results[
        "hybrid_pd"
    ]
    >
    results[
        "fdir_pd"
    ]
)


hybrid_equal_to_fdir = np.isclose(
    results[
        "hybrid_pd"
    ],
    results[
        "fdir_pd"
    ]
)


hybrid_worse_than_fdir = (
    results[
        "hybrid_pd"
    ]
    <
    results[
        "fdir_pd"
    ]
)


fraction_hybrid_better = (
    100.0
    *
    np.mean(
        hybrid_better_than_fdir
    )
)


fraction_hybrid_equal = (
    100.0
    *
    np.mean(
        hybrid_equal_to_fdir
    )
)


fraction_hybrid_worse = (
    100.0
    *
    np.mean(
        hybrid_worse_than_fdir
    )
)


# ------------------------------------------------------------
# 13. EPISODE DETECTION
# ------------------------------------------------------------

fdir_episode_detection_rate = (
    100.0
    *
    np.mean(
        results[
            "fdir_episode_detected"
        ]
    )
)


ai_episode_detection_rate = (
    100.0
    *
    np.mean(
        results[
            "temporal_ai_episode_detected"
        ]
    )
)


hybrid_episode_detection_rate = (
    100.0
    *
    np.mean(
        results[
            "hybrid_episode_detected"
        ]
    )
)


hybrid_preexisting_alarm_rate = (
    100.0
    *
    np.mean(
        results[
            "hybrid_preexisting_alarm"
        ]
    )
)


# ------------------------------------------------------------
# 14. CONDITIONAL LATENCIES
# ------------------------------------------------------------

def finite_mean(
    series
):

    values = series.to_numpy(
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

        return np.nan


    return float(
        np.mean(
            values
        )
    )


fdir_mean_latency = finite_mean(
    results[
        "fdir_latency_s"
    ]
)


ai_mean_latency = finite_mean(
    results[
        "temporal_ai_latency_s"
    ]
)


hybrid_mean_latency = finite_mean(
    results[
        "hybrid_latency_s"
    ]
)


# ------------------------------------------------------------
# 15. FAULT-MAGNITUDE QUARTILES
# ------------------------------------------------------------

results[
    "fault_magnitude_quartile"
] = pd.qcut(
    results[
        "fault_magnitude_m"
    ],
    q=4,
    labels=[
        "Q1 weakest",
        "Q2",
        "Q3",
        "Q4 strongest"
    ]
)


fault_quartiles = (
    results.groupby(
        "fault_magnitude_quartile",
        observed=True,
        as_index=False
    )
    .agg(
        scenarios=(
            "scenario_id",
            "size"
        ),

        fault_min_m=(
            "fault_magnitude_m",
            "min"
        ),

        fault_mean_m=(
            "fault_magnitude_m",
            "mean"
        ),

        fault_max_m=(
            "fault_magnitude_m",
            "max"
        ),

        fdir_pd_mean=(
            "fdir_pd",
            "mean"
        ),

        ai_pd_mean=(
            "temporal_ai_pd",
            "mean"
        ),

        hybrid_pd_mean=(
            "hybrid_pd",
            "mean"
        ),

        hybrid_gain_mean=(
            "hybrid_minus_fdir_pd",
            "mean"
        )
    )
)


fault_quartiles.to_csv(
    fault_quartile_path,
    index=False
)


# ------------------------------------------------------------
# 16. PSEUDORANGE-NOISE QUARTILES
# ------------------------------------------------------------

results[
    "noise_quartile"
] = pd.qcut(
    results[
        "pseudorange_noise_std_m"
    ],
    q=4,
    labels=[
        "Q1 lowest noise",
        "Q2",
        "Q3",
        "Q4 highest noise"
    ]
)


noise_quartiles = (
    results.groupby(
        "noise_quartile",
        observed=True,
        as_index=False
    )
    .agg(
        scenarios=(
            "scenario_id",
            "size"
        ),

        noise_min_m=(
            "pseudorange_noise_std_m",
            "min"
        ),

        noise_mean_m=(
            "pseudorange_noise_std_m",
            "mean"
        ),

        noise_max_m=(
            "pseudorange_noise_std_m",
            "max"
        ),

        fdir_pfa_mean=(
            "fdir_pfa",
            "mean"
        ),

        ai_pfa_mean=(
            "temporal_ai_pfa",
            "mean"
        ),

        hybrid_pfa_mean=(
            "hybrid_pfa",
            "mean"
        ),

        fdir_pd_mean=(
            "fdir_pd",
            "mean"
        ),

        ai_pd_mean=(
            "temporal_ai_pd",
            "mean"
        ),

        hybrid_pd_mean=(
            "hybrid_pd",
            "mean"
        )
    )
)


noise_quartiles.to_csv(
    noise_quartile_path,
    index=False
)


# ------------------------------------------------------------
# 17. SUMMARY TABLE
# ------------------------------------------------------------

summary = pd.DataFrame(
    [
        {
            "metric":
                "fdir_pd",

            "mean":
                fdir_pd_mean,

            "std":
                fdir_pd_std,

            "ci95_lower":
                fdir_pd_lower,

            "ci95_upper":
                fdir_pd_upper
        },

        {
            "metric":
                "temporal_ai_pd",

            "mean":
                ai_pd_mean,

            "std":
                ai_pd_std,

            "ci95_lower":
                ai_pd_lower,

            "ci95_upper":
                ai_pd_upper
        },

        {
            "metric":
                "hybrid_pd",

            "mean":
                hybrid_pd_mean,

            "std":
                hybrid_pd_std,

            "ci95_lower":
                hybrid_pd_lower,

            "ci95_upper":
                hybrid_pd_upper
        },

        {
            "metric":
                "hybrid_minus_fdir_pd",

            "mean":
                paired_gain_mean,

            "std":
                paired_gain_std,

            "ci95_lower":
                paired_gain_lower,

            "ci95_upper":
                paired_gain_upper
        },

        {
            "metric":
                "temporal_ai_pfa",

            "mean":
                ai_pfa_mean,

            "std":
                ai_pfa_std,

            "ci95_lower":
                ai_pfa_lower,

            "ci95_upper":
                ai_pfa_upper
        },

        {
            "metric":
                "hybrid_pfa",

            "mean":
                hybrid_pfa_mean,

            "std":
                hybrid_pfa_std,

            "ci95_lower":
                hybrid_pfa_lower,

            "ci95_upper":
                hybrid_pfa_upper
        }
    ]
)


summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 18. SCIENTIFIC FLAGS
#
# These are findings.
# They do NOT define methodological validity.
# ------------------------------------------------------------

global_ai_gain_positive = (
    paired_gain_mean
    >
    0.0
)


global_ai_gain_ci_positive = (
    paired_gain_lower
    >
    0.0
)


hybrid_mean_pfa_below_5 = (
    hybrid_pfa_mean
    <=
    5.0
)


hybrid_pfa_ci_upper_below_5 = (
    hybrid_pfa_upper
    <=
    5.0
)


hybrid_better_in_majority = (
    fraction_hybrid_better
    >
    50.0
)


# ------------------------------------------------------------
# 19. EXPERIMENT VALIDATION
#
# We deliberately DO NOT force the frozen Phase-15 model
# to remain high-performing under this wider distribution.
#
# If P_FA becomes large, that is an important robustness
# result, not an invalid experiment.
# ------------------------------------------------------------

validation_016c = (
    all_scenarios_complete
    and
    scenario_indices_complete
    and
    time_sync_valid
    and
    paired_sequences_complete
    and
    frozen_threshold_identical
    and
    core_metrics_finite
)


# ------------------------------------------------------------
# 20. SAVE FINAL RESULTS WITH QUARTILE LABELS
# ------------------------------------------------------------

results.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# 21. PRINT
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)


print(
    "AURORA — Experience 016-C RESULTS"
)


print(
    "Frozen AI / FDIR robustness campaign"
)


print(
    "========================================================================================================================"
)


print(
    f"Scenarios completed : "
    f"{len(results)}/{number_of_scenarios}"
)


print(
    f"Frozen AI threshold : "
    f"{frozen_threshold:.6f}"
)


print()


# ------------------------------------------------------------
# 22. GLOBAL DETECTION
# ------------------------------------------------------------

print(
    "----- GLOBAL EPOCH DETECTION -----"
)


print(
    f"Classical FDIR P_D : "
    f"{fdir_pd_mean:.2f} +/- "
    f"{fdir_pd_std:.2f} %"
)


print(
    f"Temporal AI P_D : "
    f"{ai_pd_mean:.2f} +/- "
    f"{ai_pd_std:.2f} %"
)


print(
    f"Persistence hybrid P_D : "
    f"{hybrid_pd_mean:.2f} +/- "
    f"{hybrid_pd_std:.2f} %"
)


print()


print(
    f"Paired hybrid - FDIR gain : "
    f"{paired_gain_mean:+.2f} percentage points"
)


print(
    f"95% CI paired gain : "
    f"[{paired_gain_lower:+.2f}, "
    f"{paired_gain_upper:+.2f}] points"
)


print()


# ------------------------------------------------------------
# 23. FALSE ALARM ROBUSTNESS
# ------------------------------------------------------------

print(
    "----- FALSE-ALARM ROBUSTNESS -----"
)


print(
    f"Classical FDIR P_FA : "
    f"{fdir_pfa_mean:.2f} +/- "
    f"{fdir_pfa_std:.2f} %"
)


print(
    f"Temporal AI P_FA : "
    f"{ai_pfa_mean:.2f} +/- "
    f"{ai_pfa_std:.2f} %"
)


print(
    f"Temporal AI P_FA 95% CI : "
    f"[{ai_pfa_lower:.2f}, "
    f"{ai_pfa_upper:.2f}] %"
)


print(
    f"Persistence hybrid P_FA : "
    f"{hybrid_pfa_mean:.2f} +/- "
    f"{hybrid_pfa_std:.2f} %"
)


print(
    f"Persistence hybrid P_FA 95% CI : "
    f"[{hybrid_pfa_lower:.2f}, "
    f"{hybrid_pfa_upper:.2f}] %"
)


print()


# ------------------------------------------------------------
# 24. SCENARIO-WISE
# ------------------------------------------------------------

print(
    "----- SCENARIO-WISE COMPARISON -----"
)


print(
    f"Hybrid better than FDIR : "
    f"{fraction_hybrid_better:.2f} %"
)


print(
    f"Hybrid equal to FDIR : "
    f"{fraction_hybrid_equal:.2f} %"
)


print(
    f"Hybrid worse than FDIR : "
    f"{fraction_hybrid_worse:.2f} %"
)


print()


# ------------------------------------------------------------
# 25. EPISODE DETECTION
# ------------------------------------------------------------

print(
    "----- FAULT-EPISODE DETECTION -----"
)


print(
    f"Classical FDIR : "
    f"{fdir_episode_detection_rate:.2f} %"
)


print(
    f"Temporal AI : "
    f"{ai_episode_detection_rate:.2f} %"
)


print(
    f"Persistence hybrid : "
    f"{hybrid_episode_detection_rate:.2f} %"
)


print(
    f"Hybrid already in alarm before fault onset : "
    f"{hybrid_preexisting_alarm_rate:.2f} % of scenarios"
)


print()


print(
    "Conditional mean latency among NEW detected episodes:"
)


print(
    f"FDIR : "
    f"{fdir_mean_latency:.1f} s"
)


print(
    f"Temporal AI : "
    f"{ai_mean_latency:.1f} s"
)


print(
    f"Hybrid : "
    f"{hybrid_mean_latency:.1f} s"
)


print()


# ------------------------------------------------------------
# 26. FAULT QUARTILES
# ------------------------------------------------------------

print(
    "----- PERFORMANCE BY FAULT-MAGNITUDE QUARTILE -----"
)


print(
    fault_quartiles.to_string(
        index=False,
        formatters={
            "fault_min_m":
                lambda value:
                    f"{value:.1f}",

            "fault_mean_m":
                lambda value:
                    f"{value:.1f}",

            "fault_max_m":
                lambda value:
                    f"{value:.1f}",

            "fdir_pd_mean":
                lambda value:
                    f"{value:.2f}",

            "ai_pd_mean":
                lambda value:
                    f"{value:.2f}",

            "hybrid_pd_mean":
                lambda value:
                    f"{value:.2f}",

            "hybrid_gain_mean":
                lambda value:
                    f"{value:+.2f}"
        }
    )
)


print()


# ------------------------------------------------------------
# 27. NOISE QUARTILES
# ------------------------------------------------------------

print(
    "----- PERFORMANCE BY PSEUDORANGE-NOISE QUARTILE -----"
)


print(
    noise_quartiles.to_string(
        index=False,
        formatters={
            "noise_min_m":
                lambda value:
                    f"{value:.2f}",

            "noise_mean_m":
                lambda value:
                    f"{value:.2f}",

            "noise_max_m":
                lambda value:
                    f"{value:.2f}",

            "fdir_pfa_mean":
                lambda value:
                    f"{value:.2f}",

            "ai_pfa_mean":
                lambda value:
                    f"{value:.2f}",

            "hybrid_pfa_mean":
                lambda value:
                    f"{value:.2f}",

            "fdir_pd_mean":
                lambda value:
                    f"{value:.2f}",

            "ai_pd_mean":
                lambda value:
                    f"{value:.2f}",

            "hybrid_pd_mean":
                lambda value:
                    f"{value:.2f}"
        }
    )
)


print()


# ------------------------------------------------------------
# 28. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Mean hybrid gain > 0 : "
    f"{global_ai_gain_positive}"
)


print(
    f"95% CI of mean hybrid gain entirely > 0 : "
    f"{global_ai_gain_ci_positive}"
)


print(
    f"Hybrid better than FDIR in majority of scenarios : "
    f"{hybrid_better_in_majority}"
)


print(
    f"Hybrid mean P_FA <= 5% : "
    f"{hybrid_mean_pfa_below_5}"
)


print(
    f"Hybrid P_FA CI upper bound <= 5% : "
    f"{hybrid_pfa_ci_upper_below_5}"
)


print()


# ------------------------------------------------------------
# 29. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"All 64 scenarios complete : "
    f"{all_scenarios_complete}"
)


print(
    f"Scenario indices complete : "
    f"{scenario_indices_complete}"
)


print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)


print(
    f"Paired nominal/fault sequences complete : "
    f"{paired_sequences_complete}"
)


print(
    f"Frozen threshold identical in every scenario : "
    f"{frozen_threshold_identical}"
)


print(
    f"All core metrics finite : "
    f"{core_metrics_finite}"
)


print()


print(
    f"VALIDATION GLOBALE 016-C : "
    f"{validation_016c}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 30. FIGURE — FAULT MAGNITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    results[
        "fault_magnitude_m"
    ],

    results[
        "fdir_pd"
    ],

    label="Classical FDIR"
)


plt.scatter(
    results[
        "fault_magnitude_m"
    ],

    results[
        "hybrid_pd"
    ],

    label="Persistence hybrid"
)


plt.xlabel(
    "Fault magnitude [m]"
)


plt.ylabel(
    "Epoch detection rate [%]"
)


plt.title(
    "AURORA — 016-C detection robustness vs fault magnitude"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016c_detection_vs_fault.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 31. FIGURE — FALSE ALARM VS NOISE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    results[
        "pseudorange_noise_std_m"
    ],

    results[
        "fdir_pfa"
    ],

    label="Classical FDIR"
)


plt.scatter(
    results[
        "pseudorange_noise_std_m"
    ],

    results[
        "temporal_ai_pfa"
    ],

    label="Temporal AI"
)


plt.scatter(
    results[
        "pseudorange_noise_std_m"
    ],

    results[
        "hybrid_pfa"
    ],

    label="Persistence hybrid"
)


plt.xlabel(
    "Pseudorange noise std [m]"
)


plt.ylabel(
    "Nominal false-alarm rate [%]"
)


plt.title(
    "AURORA — 016-C false-alarm robustness vs GNSS noise"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016c_false_alarm_vs_noise.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 32. FIGURE — PAIRED DETECTION
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)


plt.scatter(
    results[
        "fdir_pd"
    ],

    results[
        "hybrid_pd"
    ]
)


plt.plot(
    [
        0.0,
        100.0
    ],

    [
        0.0,
        100.0
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
    "AURORA — 016-C paired robustness comparison"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016c_paired_detection.png",

    dpi=200
)


plt.show()