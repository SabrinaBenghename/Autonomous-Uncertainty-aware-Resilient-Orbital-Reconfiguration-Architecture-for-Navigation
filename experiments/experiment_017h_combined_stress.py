from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    spearmanr,
    wilcoxon,
)

import src.validation.outage_phase_sweep as outage_module


# ============================================================
# AURORA — EXPERIMENT 017-H
# COMBINED MATURITY x OUTAGE-DURATION STRESS TEST
#
# 017-G identifies the empirically hardest tested outage
# duration for a mature estimator.
#
# 017-H then compares:
#
#   estimator ages = [5, 30, 60] min
#
# at:
#
#   reference duration = 15 min
#   stress duration    = worst duration found by 017-G
#
# All conditions use the same outage start (60 min), phase,
# paired seeds, sensors, and physical trajectory.
#
# 3 ages x 2 durations x 8 replicates = 48 missions
#
# The campaign is checkpoint-safe.
# ============================================================


# ------------------------------------------------------------
# 1. INPUT FROM 017-G
# ------------------------------------------------------------

table_directory = Path("results") / "tables"
figure_directory = Path("results") / "figures"
data_directory = Path("data") / "phase17"

table_directory.mkdir(parents=True, exist_ok=True)
figure_directory.mkdir(parents=True, exist_ok=True)
data_directory.mkdir(parents=True, exist_ok=True)

g_summary_path = (
    table_directory
    / "phase17_017g_duration_summary.csv"
)

if not g_summary_path.exists():
    raise RuntimeError(
        "017-H aborted before simulation: "
        "run 017-G first. Missing "
        "phase17_017g_duration_summary.csv"
    )

g_summary = pd.read_csv(
    g_summary_path
)

required_g_columns = [
    "experiment_outage_duration_min",
    "outage_rmse_mean_m",
    "outage_end_error_mean_m",
]

if not all(
    column in g_summary.columns
    for column in required_g_columns
):
    raise RuntimeError(
        "017-G summary does not contain expected columns."
    )


# ------------------------------------------------------------
# 2. CHOOSE STRESS DURATION
#
# Primary selector:
# maximum mean end-of-outage error.
#
# If 15 min were somehow selected, use the non-15 duration
# with the largest mean end error so that H has two distinct
# duration levels.
# ------------------------------------------------------------

REFERENCE_DURATION_MIN = 15.0

ranked_g = g_summary.sort_values(
    "outage_end_error_mean_m",
    ascending=False,
).reset_index(drop=True)

STRESS_DURATION_MIN = float(
    ranked_g.loc[
        0,
        "experiment_outage_duration_min",
    ]
)

if np.isclose(
    STRESS_DURATION_MIN,
    REFERENCE_DURATION_MIN,
):
    alternatives = ranked_g[
        ~np.isclose(
            ranked_g[
                "experiment_outage_duration_min"
            ].astype(float),
            REFERENCE_DURATION_MIN,
        )
    ]

    if len(alternatives) == 0:
        raise RuntimeError(
            "017-H requires at least two distinct durations."
        )

    STRESS_DURATION_MIN = float(
        alternatives.iloc[
            0
        ][
            "experiment_outage_duration_min"
        ]
    )


# ------------------------------------------------------------
# 3. DESIGN
# ------------------------------------------------------------

FIXED_INITIAL_TRUE_ANOMALY_DEG = 0.0
FIXED_OUTAGE_START_MIN = 60.0

ESTIMATOR_AGES_MIN = [
    5.0,
    30.0,
    60.0,
]

OUTAGE_DURATIONS_MIN = [
    REFERENCE_DURATION_MIN,
    STRESS_DURATION_MIN,
]

POST_RECOVERY_WINDOW_MIN = 15.0

NUMBER_OF_REPLICATES = 8
BASE_SEED = 170_700

design_path = (
    data_directory
    / "combined_stress_design_017h.csv"
)
results_path = (
    table_directory
    / "phase17_017h_combined_stress.csv"
)
summary_path = (
    table_directory
    / "phase17_017h_combined_stress_summary.csv"
)
age_statistics_path = (
    table_directory
    / "phase17_017h_age_friedman_statistics.csv"
)
duration_statistics_path = (
    table_directory
    / "phase17_017h_duration_wilcoxon_statistics.csv"
)
interaction_path = (
    table_directory
    / "phase17_017h_interaction_penalty_statistics.csv"
)
correlation_path = (
    table_directory
    / "phase17_017h_correlations.csv"
)
validation_path = (
    table_directory
    / "phase17_017h_validation.csv"
)


# ------------------------------------------------------------
# 4. INTERFACE CHECK
# ------------------------------------------------------------

run_function = outage_module.run_dual_outage_phase_scenario

if "estimator_start_min" not in run_function.__code__.co_varnames:
    raise RuntimeError(
        "017-H aborted before simulation: "
        "estimator_start_min control is unavailable."
    )


# ------------------------------------------------------------
# 5. RUN ONE CONDITION
# ------------------------------------------------------------

def run_combined_scenario(
    estimator_age_min,
    outage_duration_min,
    replicate_index,
    scenario_seed,
):
    estimator_age_min = float(
        estimator_age_min
    )
    outage_duration_min = float(
        outage_duration_min
    )

    estimator_start_min = (
        FIXED_OUTAGE_START_MIN
        -
        estimator_age_min
    )

    if estimator_start_min < 0.0:
        raise ValueError(
            "Estimator age exceeds outage-start time."
        )

    outage_end_min = (
        FIXED_OUTAGE_START_MIN
        +
        outage_duration_min
    )

    post_recovery_end_min = (
        outage_end_min
        +
        POST_RECOVERY_WINDOW_MIN
    )

    old_start = getattr(
        outage_module,
        "DUAL_OUTAGE_START_MIN",
        None,
    )
    old_duration = getattr(
        outage_module,
        "DUAL_OUTAGE_DURATION_MIN",
        None,
    )
    old_end = getattr(
        outage_module,
        "DUAL_OUTAGE_END_MIN",
        None,
    )
    old_post = getattr(
        outage_module,
        "POST_RECOVERY_END_MIN",
        None,
    )

    try:
        outage_module.DUAL_OUTAGE_START_MIN = (
            FIXED_OUTAGE_START_MIN
        )
        outage_module.DUAL_OUTAGE_DURATION_MIN = (
            outage_duration_min
        )
        outage_module.DUAL_OUTAGE_END_MIN = (
            outage_end_min
        )
        outage_module.POST_RECOVERY_END_MIN = (
            post_recovery_end_min
        )

        result = run_function(
            initial_true_anomaly_deg=
                FIXED_INITIAL_TRUE_ANOMALY_DEG,
            replicate_index=
                int(replicate_index),
            scenario_seed=
                int(scenario_seed),
            estimator_start_min=
                float(estimator_start_min),
        )

    finally:
        if old_start is not None:
            outage_module.DUAL_OUTAGE_START_MIN = old_start
        if old_duration is not None:
            outage_module.DUAL_OUTAGE_DURATION_MIN = old_duration
        if old_end is not None:
            outage_module.DUAL_OUTAGE_END_MIN = old_end
        if old_post is not None:
            outage_module.POST_RECOVERY_END_MIN = old_post

    result = dict(result)

    result["experiment_estimator_age_min"] = (
        estimator_age_min
    )
    result["experiment_estimator_start_min"] = (
        estimator_start_min
    )
    result["experiment_outage_start_min"] = (
        FIXED_OUTAGE_START_MIN
    )
    result["experiment_outage_duration_min"] = (
        outage_duration_min
    )
    result["experiment_outage_end_min"] = (
        outage_end_min
    )
    result["experiment_post_recovery_end_min"] = (
        post_recovery_end_min
    )
    result["fixed_initial_true_anomaly_deg"] = (
        FIXED_INITIAL_TRUE_ANOMALY_DEG
    )

    return result


# ------------------------------------------------------------
# 6. HEADER
# ------------------------------------------------------------

total_expected_runs = (
    len(ESTIMATOR_AGES_MIN)
    *
    len(OUTAGE_DURATIONS_MIN)
    *
    NUMBER_OF_REPLICATES
)

print()
print("=" * 120)
print("AURORA — EXPERIMENT 017-H")
print("Combined estimator-maturity x outage-duration stress test")
print("=" * 120)
print(
    f"017-G selected stress duration : "
    f"{STRESS_DURATION_MIN:.1f} min"
)
print(
    f"Duration levels : "
    f"{OUTAGE_DURATIONS_MIN} min"
)
print(
    f"Estimator ages : "
    f"{ESTIMATOR_AGES_MIN} min"
)
print(
    f"Fixed outage start : "
    f"{FIXED_OUTAGE_START_MIN:.1f} min"
)
print(
    f"Replicates per condition : "
    f"{NUMBER_OF_REPLICATES}"
)
print(
    f"Total expected missions : "
    f"{total_expected_runs}"
)
print()


# ------------------------------------------------------------
# 7. DESIGN TABLE
# ------------------------------------------------------------

design_rows = []

for replicate_index in range(
    NUMBER_OF_REPLICATES
):
    scenario_seed = (
        BASE_SEED
        +
        replicate_index
    )

    for duration_min in (
        OUTAGE_DURATIONS_MIN
    ):
        for estimator_age_min in (
            ESTIMATOR_AGES_MIN
        ):
            design_rows.append(
                {
                    "estimator_age_min":
                        float(
                            estimator_age_min
                        ),
                    "estimator_start_min":
                        float(
                            FIXED_OUTAGE_START_MIN
                            -
                            estimator_age_min
                        ),
                    "outage_duration_min":
                        float(
                            duration_min
                        ),
                    "outage_start_min":
                        FIXED_OUTAGE_START_MIN,
                    "outage_end_min":
                        FIXED_OUTAGE_START_MIN
                        +
                        float(
                            duration_min
                        ),
                    "replicate_index":
                        int(
                            replicate_index
                        ),
                    "scenario_seed":
                        int(
                            scenario_seed
                        ),
                }
            )

design = pd.DataFrame(
    design_rows
)

design.to_csv(
    design_path,
    index=False,
)

design_valid = bool(
    len(design)
    ==
    total_expected_runs
)


# ------------------------------------------------------------
# 8. CHECKPOINT
# ------------------------------------------------------------

if results_path.exists():
    existing_results = pd.read_csv(
        results_path
    )

    required_keys = [
        "experiment_estimator_age_min",
        "experiment_outage_duration_min",
        "replicate_index",
    ]

    if not all(
        column in existing_results.columns
        for column in required_keys
    ):
        raise RuntimeError(
            "Existing 017-H checkpoint is incompatible."
        )

    existing_results = (
        existing_results
        .sort_values(
            [
                "replicate_index",
                "experiment_outage_duration_min",
                "experiment_estimator_age_min",
            ]
        )
        .drop_duplicates(
            subset=[
                "experiment_estimator_age_min",
                "experiment_outage_duration_min",
                "replicate_index",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    existing_results.to_csv(
        results_path,
        index=False,
    )

    completed_keys = set(
        zip(
            existing_results[
                "experiment_estimator_age_min"
            ].astype(float),
            existing_results[
                "experiment_outage_duration_min"
            ].astype(float),
            existing_results[
                "replicate_index"
            ].astype(int),
        )
    )

else:
    existing_results = pd.DataFrame()
    completed_keys = set()


remaining_rows = []

for _, row in design.iterrows():
    key = (
        float(
            row[
                "estimator_age_min"
            ]
        ),
        float(
            row[
                "outage_duration_min"
            ]
        ),
        int(
            row[
                "replicate_index"
            ]
        ),
    )

    if key not in completed_keys:
        remaining_rows.append(
            row
        )

remaining_design = pd.DataFrame(
    remaining_rows
)

print(
    f"Already completed : "
    f"{len(completed_keys)}"
)
print(
    f"Remaining : "
    f"{len(remaining_design)}"
)

if len(
    remaining_design
) == 0:
    print(
        "Checkpoint complete — "
        "NO 017-H missions will be repeated."
    )

print()


# ------------------------------------------------------------
# 9. RUN
# ------------------------------------------------------------

new_rows = []

for local_index, (_, row) in enumerate(
    remaining_design.iterrows()
):
    age_min = float(
        row[
            "estimator_age_min"
        ]
    )
    duration_min = float(
        row[
            "outage_duration_min"
        ]
    )
    replicate_index = int(
        row[
            "replicate_index"
        ]
    )
    scenario_seed = int(
        row[
            "scenario_seed"
        ]
    )

    global_index = (
        len(completed_keys)
        +
        local_index
        +
        1
    )

    print(
        f"[{global_index:02d}/{total_expected_runs}] "
        f"age={age_min:5.1f} min | "
        f"duration={duration_min:5.1f} min | "
        f"rep={replicate_index:02d} | "
        f"seed={scenario_seed}"
    )

    result = run_combined_scenario(
        estimator_age_min=
            age_min,
        outage_duration_min=
            duration_min,
        replicate_index=
            replicate_index,
        scenario_seed=
            scenario_seed,
    )

    new_rows.append(
        result
    )

    if len(
        existing_results
    ) > 0:
        checkpoint = pd.concat(
            [
                existing_results,
                pd.DataFrame(
                    new_rows
                ),
            ],
            ignore_index=True,
        )
    else:
        checkpoint = pd.DataFrame(
            new_rows
        )

    checkpoint = (
        checkpoint
        .sort_values(
            [
                "replicate_index",
                "experiment_outage_duration_min",
                "experiment_estimator_age_min",
            ]
        )
        .drop_duplicates(
            subset=[
                "experiment_estimator_age_min",
                "experiment_outage_duration_min",
                "replicate_index",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    checkpoint.to_csv(
        results_path,
        index=False,
    )

    print(
        f"    sigma_start="
        f"{result['outage_start_sigma_m']:.4f} m | "
        f"RMSE="
        f"{result['outage_rmse_m']:.3f} m | "
        f"end="
        f"{result['outage_end_error_m']:.3f} m | "
        f"sigma_end="
        f"{result['outage_end_sigma_m']:.3f} m | "
        f"post="
        f"{result['post_recovery_rmse_m']:.3f} m"
    )


# ------------------------------------------------------------
# 10. FINAL DATA
# ------------------------------------------------------------

results = pd.read_csv(
    results_path
)

results = (
    results
    .sort_values(
        [
            "replicate_index",
            "experiment_outage_duration_min",
            "experiment_estimator_age_min",
        ]
    )
    .drop_duplicates(
        subset=[
            "experiment_estimator_age_min",
            "experiment_outage_duration_min",
            "replicate_index",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 11. SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        [
            "experiment_estimator_age_min",
            "experiment_outage_duration_min",
        ],
        as_index=False,
    )
    .agg(
        runs=(
            "replicate_index",
            "size",
        ),
        sigma_start_mean_m=(
            "outage_start_sigma_m",
            "mean",
        ),
        outage_rmse_mean_m=(
            "outage_rmse_m",
            "mean",
        ),
        outage_rmse_std_m=(
            "outage_rmse_m",
            "std",
        ),
        outage_end_error_mean_m=(
            "outage_end_error_m",
            "mean",
        ),
        outage_end_error_std_m=(
            "outage_end_error_m",
            "std",
        ),
        sigma_end_mean_m=(
            "outage_end_sigma_m",
            "mean",
        ),
        post_recovery_rmse_mean_m=(
            "post_recovery_rmse_m",
            "mean",
        ),
        navigation_nis_mean=(
            "navigation_nis_mean",
            "mean",
        ),
    )
)

summary.to_csv(
    summary_path,
    index=False,
)


# ------------------------------------------------------------
# 12. AGE EFFECT WITHIN EACH DURATION
# ------------------------------------------------------------

age_stat_rows = []

for duration_min in OUTAGE_DURATIONS_MIN:
    block = results[
        np.isclose(
            results[
                "experiment_outage_duration_min"
            ].astype(float),
            duration_min,
        )
    ]

    for metric in [
        "outage_rmse_m",
        "outage_end_error_m",
        "outage_end_sigma_m",
    ]:
        pivot = (
            block.pivot(
                index="replicate_index",
                columns="experiment_estimator_age_min",
                values=metric,
            )
            .reindex(
                columns=ESTIMATOR_AGES_MIN
            )
        )

        samples = [
            pivot[
                age
            ].to_numpy(dtype=float)
            for age in ESTIMATOR_AGES_MIN
        ]

        statistic, p_value = (
            friedmanchisquare(
                *samples
            )
        )

        kendalls_w = float(
            statistic
            /
            (
                NUMBER_OF_REPLICATES
                *
                (
                    len(
                        ESTIMATOR_AGES_MIN
                    )
                    -
                    1
                )
            )
        )

        age_stat_rows.append(
            {
                "outage_duration_min":
                    float(duration_min),
                "metric":
                    metric,
                "friedman_statistic":
                    float(statistic),
                "p_value":
                    float(p_value),
                "kendalls_w":
                    kendalls_w,
                "significant_0p05":
                    bool(
                        p_value < 0.05
                    ),
            }
        )

age_statistics = pd.DataFrame(
    age_stat_rows
)

age_statistics.to_csv(
    age_statistics_path,
    index=False,
)


# ------------------------------------------------------------
# 13. DURATION EFFECT WITHIN EACH AGE
#
# Only two duration levels -> paired Wilcoxon.
# ------------------------------------------------------------

duration_rows = []

for age_min in ESTIMATOR_AGES_MIN:
    block = results[
        np.isclose(
            results[
                "experiment_estimator_age_min"
            ].astype(float),
            age_min,
        )
    ]

    for metric in [
        "outage_rmse_m",
        "outage_end_error_m",
        "outage_end_sigma_m",
        "post_recovery_rmse_m",
    ]:
        pivot = (
            block.pivot(
                index="replicate_index",
                columns="experiment_outage_duration_min",
                values=metric,
            )
            .reindex(
                columns=OUTAGE_DURATIONS_MIN
            )
        )

        reference = pivot[
            REFERENCE_DURATION_MIN
        ].to_numpy(dtype=float)

        stress = pivot[
            STRESS_DURATION_MIN
        ].to_numpy(dtype=float)

        statistic, p_value = wilcoxon(
            reference,
            stress,
            alternative="two-sided",
        )

        duration_rows.append(
            {
                "estimator_age_min":
                    float(age_min),
                "metric":
                    metric,
                "reference_duration_min":
                    REFERENCE_DURATION_MIN,
                "stress_duration_min":
                    STRESS_DURATION_MIN,
                "reference_mean":
                    float(
                        np.mean(reference)
                    ),
                "stress_mean":
                    float(
                        np.mean(stress)
                    ),
                "mean_stress_minus_reference":
                    float(
                        np.mean(
                            stress
                            -
                            reference
                        )
                    ),
                "wilcoxon_statistic":
                    float(statistic),
                "p_value":
                    float(p_value),
                "significant_0p05":
                    bool(
                        p_value < 0.05
                    ),
            }
        )

duration_statistics = pd.DataFrame(
    duration_rows
)

duration_statistics.to_csv(
    duration_statistics_path,
    index=False,
)


# ------------------------------------------------------------
# 14. AGE x DURATION INTERACTION DIAGNOSTIC
#
# For each replicate and each age:
#
#   penalty =
#       stress-duration metric
#       -
#       reference-duration metric
#
# If the penalty changes significantly with age, the effect
# of a long outage depends on estimator maturity.
# ------------------------------------------------------------

interaction_rows = []

for metric in [
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
]:
    penalty_by_age = {}

    for age_min in ESTIMATOR_AGES_MIN:
        block = results[
            np.isclose(
                results[
                    "experiment_estimator_age_min"
                ].astype(float),
                age_min,
            )
        ]

        pivot = (
            block.pivot(
                index="replicate_index",
                columns="experiment_outage_duration_min",
                values=metric,
            )
            .reindex(
                columns=OUTAGE_DURATIONS_MIN
            )
        )

        penalty_by_age[
            age_min
        ] = (
            pivot[
                STRESS_DURATION_MIN
            ].to_numpy(dtype=float)
            -
            pivot[
                REFERENCE_DURATION_MIN
            ].to_numpy(dtype=float)
        )

    samples = [
        penalty_by_age[
            age
        ]
        for age in ESTIMATOR_AGES_MIN
    ]

    statistic, p_value = friedmanchisquare(
        *samples
    )

    kendalls_w = float(
        statistic
        /
        (
            NUMBER_OF_REPLICATES
            *
            (
                len(
                    ESTIMATOR_AGES_MIN
                )
                -
                1
            )
        )
    )

    interaction_rows.append(
        {
            "metric":
                metric,
            "reference_duration_min":
                REFERENCE_DURATION_MIN,
            "stress_duration_min":
                STRESS_DURATION_MIN,
            "age_5_mean_penalty":
                float(
                    np.mean(
                        penalty_by_age[
                            5.0
                        ]
                    )
                ),
            "age_30_mean_penalty":
                float(
                    np.mean(
                        penalty_by_age[
                            30.0
                        ]
                    )
                ),
            "age_60_mean_penalty":
                float(
                    np.mean(
                        penalty_by_age[
                            60.0
                        ]
                    )
                ),
            "friedman_statistic":
                float(statistic),
            "p_value":
                float(p_value),
            "kendalls_w":
                kendalls_w,
            "interaction_evidence_0p05":
                bool(
                    p_value < 0.05
                ),
        }
    )

interaction_table = pd.DataFrame(
    interaction_rows
)

interaction_table.to_csv(
    interaction_path,
    index=False,
)


# ------------------------------------------------------------
# 15. CORRELATIONS
# ------------------------------------------------------------

correlation_rows = []

for predictor in [
    "experiment_estimator_age_min",
    "experiment_outage_duration_min",
    "outage_start_sigma_m",
]:
    for outcome in [
        "outage_rmse_m",
        "outage_end_error_m",
    ]:
        x = results[
            predictor
        ].to_numpy(dtype=float)

        y = results[
            outcome
        ].to_numpy(dtype=float)

        rho, p_value = spearmanr(
            x,
            y,
        )

        correlation_rows.append(
            {
                "predictor":
                    predictor,
                "outcome":
                    outcome,
                "n":
                    int(len(results)),
                "spearman_rho":
                    float(rho),
                "p_value":
                    float(p_value),
                "significant_0p05":
                    bool(
                        p_value < 0.05
                    ),
            }
        )

correlation_table = pd.DataFrame(
    correlation_rows
)

correlation_table.to_csv(
    correlation_path,
    index=False,
)


# ------------------------------------------------------------
# 16. VALIDATION
# ------------------------------------------------------------

all_runs_complete = bool(
    len(results)
    ==
    total_expected_runs
)

unique_keys = bool(
    results[
        [
            "experiment_estimator_age_min",
            "experiment_outage_duration_min",
            "replicate_index",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    ==
    total_expected_runs
)

all_cells_have_8 = bool(
    np.all(
        results.groupby(
            [
                "experiment_estimator_age_min",
                "experiment_outage_duration_min",
            ]
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)

fixed_start_valid = bool(
    np.allclose(
        results[
            "experiment_outage_start_min"
        ].to_numpy(dtype=float),
        FIXED_OUTAGE_START_MIN,
    )
)

age_identity_valid = bool(
    np.allclose(
        results[
            "experiment_estimator_age_min"
        ].to_numpy(dtype=float)
        +
        results[
            "experiment_estimator_start_min"
        ].to_numpy(dtype=float),
        FIXED_OUTAGE_START_MIN,
    )
)

phase_fixed_valid = bool(
    np.allclose(
        results[
            "fixed_initial_true_anomaly_deg"
        ].to_numpy(dtype=float),
        FIXED_INITIAL_TRUE_ANOMALY_DEG,
    )
)

core_columns = [
    "outage_start_sigma_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
    "post_recovery_rmse_m",
    "navigation_nis_mean",
    "attitude_outage_rmse_deg",
]

core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_columns
            ].to_numpy(dtype=float)
        )
    )
)

time_sync_valid = bool(
    np.max(
        np.abs(
            results[
                "time_sync_error_s"
            ].to_numpy(dtype=float)
        )
    )
    <
    1.0e-12
)

validation_017h = bool(
    design_valid
    and
    all_runs_complete
    and
    unique_keys
    and
    all_cells_have_8
    and
    fixed_start_valid
    and
    age_identity_valid
    and
    phase_fixed_valid
    and
    core_metrics_finite
    and
    time_sync_valid
)

pd.DataFrame(
    [
        {
            "design_valid":
                design_valid,
            "all_runs_complete":
                all_runs_complete,
            "unique_keys":
                unique_keys,
            "all_cells_have_8":
                all_cells_have_8,
            "fixed_start_valid":
                fixed_start_valid,
            "age_identity_valid":
                age_identity_valid,
            "phase_fixed_valid":
                phase_fixed_valid,
            "core_metrics_finite":
                core_metrics_finite,
            "time_sync_valid":
                time_sync_valid,
            "global_validation":
                validation_017h,
        }
    ]
).to_csv(
    validation_path,
    index=False,
)


# ------------------------------------------------------------
# 17. PRINT
# ------------------------------------------------------------

print()
print("=" * 120)
print("AURORA — EXPERIMENT 017-H RESULTS")
print("Combined maturity x outage-duration stress test")
print("=" * 120)
print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)
print(
    f"Reference duration : "
    f"{REFERENCE_DURATION_MIN:.1f} min"
)
print(
    f"017-G stress duration : "
    f"{STRESS_DURATION_MIN:.1f} min"
)
print()

print("----- COMBINED CONDITION SUMMARY -----")
print(
    summary.to_string(
        index=False,
        formatters={
            column:
                lambda value:
                    f"{value:.4f}"
            for column in summary.columns
            if column != "runs"
        },
    )
)
print()

print("----- AGE EFFECT WITHIN DURATION -----")
print(
    age_statistics.to_string(
        index=False,
        formatters={
            "friedman_statistic":
                lambda value:
                    f"{value:.3f}",
            "p_value":
                lambda value:
                    f"{value:.6f}",
            "kendalls_w":
                lambda value:
                    f"{value:.3f}",
        },
    )
)
print()

print("----- DURATION EFFECT WITHIN AGE -----")
print(
    duration_statistics.to_string(
        index=False,
        formatters={
            "reference_mean":
                lambda value:
                    f"{value:.4f}",
            "stress_mean":
                lambda value:
                    f"{value:.4f}",
            "mean_stress_minus_reference":
                lambda value:
                    f"{value:.4f}",
            "wilcoxon_statistic":
                lambda value:
                    f"{value:.3f}",
            "p_value":
                lambda value:
                    f"{value:.6f}",
        },
    )
)
print()

print("----- AGE x DURATION INTERACTION DIAGNOSTIC -----")
print(
    interaction_table.to_string(
        index=False,
        formatters={
            "age_5_mean_penalty":
                lambda value:
                    f"{value:.4f}",
            "age_30_mean_penalty":
                lambda value:
                    f"{value:.4f}",
            "age_60_mean_penalty":
                lambda value:
                    f"{value:.4f}",
            "friedman_statistic":
                lambda value:
                    f"{value:.3f}",
            "p_value":
                lambda value:
                    f"{value:.6f}",
            "kendalls_w":
                lambda value:
                    f"{value:.3f}",
        },
    )
)
print()

print("----- VALIDATION -----")
print(
    f"All {total_expected_runs} missions complete : "
    f"{all_runs_complete}"
)
print(
    f"Every age-duration cell has 8 replicates : "
    f"{all_cells_have_8}"
)
print(
    f"Outage start fixed at 60 min : "
    f"{fixed_start_valid}"
)
print(
    f"Estimator age/start identity valid : "
    f"{age_identity_valid}"
)
print(
    f"Initial phase fixed : "
    f"{phase_fixed_valid}"
)
print(
    f"All primary metrics finite : "
    f"{core_metrics_finite}"
)
print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)
print()
print(
    f"VALIDATION GLOBALE 017-H : "
    f"{validation_017h}"
)
print("=" * 120)


# ------------------------------------------------------------
# 18. FIGURES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for duration_min in OUTAGE_DURATIONS_MIN:
    block = summary[
        np.isclose(
            summary[
                "experiment_outage_duration_min"
            ].astype(float),
            duration_min,
        )
    ].sort_values(
        "experiment_estimator_age_min"
    )

    plt.plot(
        block[
            "experiment_estimator_age_min"
        ],
        block[
            "outage_rmse_mean_m"
        ],
        marker="o",
        label=
            f"{duration_min:.0f}-min outage",
    )

plt.xlabel("Estimator age at outage onset [min]")
plt.ylabel("Mean outage RMSE [m]")
plt.title(
    "AURORA — 017-H maturity x outage-duration stress"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017h_maturity_duration_rmse.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 6))

for duration_min in OUTAGE_DURATIONS_MIN:
    block = summary[
        np.isclose(
            summary[
                "experiment_outage_duration_min"
            ].astype(float),
            duration_min,
        )
    ].sort_values(
        "experiment_estimator_age_min"
    )

    plt.plot(
        block[
            "experiment_estimator_age_min"
        ],
        block[
            "outage_end_error_mean_m"
        ],
        marker="o",
        label=
            f"{duration_min:.0f}-min outage",
    )

plt.xlabel("Estimator age at outage onset [min]")
plt.ylabel("Mean end-of-outage error [m]")
plt.title(
    "AURORA — 017-H terminal-error stress response"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017h_maturity_duration_end_error.png",
    dpi=200,
)
plt.close()
