from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, spearmanr

import src.validation.outage_phase_sweep as outage_module


# ============================================================
# AURORA — EXPERIMENT 017-G
# OUTAGE-DURATION RESILIENCE SWEEP
#
# Question:
# Once the navigation estimator is mature, how does resilience
# degrade as the simultaneous GNSS + star-tracker outage lasts
# longer?
#
# FIXED:
#   initial true anomaly = 0 deg
#   outage start         = 60 min
#   estimator start      = 0 min
#   paired seed          = same across durations
#
# VARIABLE:
#   outage duration = [5, 10, 15, 20, 30] min
#
# 5 durations x 8 replicates = 40 missions
#
# Checkpoint-safe:
# completed duration/replicate pairs are never rerun.
# ============================================================


# ------------------------------------------------------------
# 1. DESIGN
# ------------------------------------------------------------

FIXED_INITIAL_TRUE_ANOMALY_DEG = 0.0
FIXED_OUTAGE_START_MIN = 60.0
FIXED_ESTIMATOR_START_MIN = 0.0

OUTAGE_DURATIONS_MIN = [
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
]

POST_RECOVERY_WINDOW_MIN = 15.0

NUMBER_OF_REPLICATES = 8
BASE_SEED = 170_600


# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

table_directory = Path("results") / "tables"
figure_directory = Path("results") / "figures"
data_directory = Path("data") / "phase17"

table_directory.mkdir(parents=True, exist_ok=True)
figure_directory.mkdir(parents=True, exist_ok=True)
data_directory.mkdir(parents=True, exist_ok=True)

design_path = data_directory / "outage_duration_design_017g.csv"
results_path = table_directory / "phase17_017g_outage_duration_sweep.csv"
summary_path = table_directory / "phase17_017g_duration_summary.csv"
statistics_path = table_directory / "phase17_017g_friedman_statistics.csv"
correlation_path = table_directory / "phase17_017g_correlations.csv"
validation_path = table_directory / "phase17_017g_validation.csv"


# ------------------------------------------------------------
# 3. INTERFACE CHECK
# ------------------------------------------------------------

run_function = outage_module.run_dual_outage_phase_scenario

if "estimator_start_min" not in run_function.__code__.co_varnames:
    raise RuntimeError(
        "017-G aborted before simulation: "
        "run_dual_outage_phase_scenario() does not expose "
        "estimator_start_min."
    )


# ------------------------------------------------------------
# 4. ONE CONTROLLED DURATION RUN
# ------------------------------------------------------------

def run_duration_scenario(
    outage_duration_min,
    replicate_index,
    scenario_seed,
):
    outage_duration_min = float(outage_duration_min)
    outage_end_min = (
        FIXED_OUTAGE_START_MIN
        + outage_duration_min
    )
    post_recovery_end_min = (
        outage_end_min
        + POST_RECOVERY_WINDOW_MIN
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
                FIXED_ESTIMATOR_START_MIN,
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

    result["experiment_outage_start_min"] = (
        FIXED_OUTAGE_START_MIN
    )
    result["experiment_outage_end_min"] = (
        outage_end_min
    )
    result["experiment_outage_duration_min"] = (
        outage_duration_min
    )
    result["experiment_post_recovery_end_min"] = (
        post_recovery_end_min
    )
    result["experiment_estimator_start_min"] = (
        FIXED_ESTIMATOR_START_MIN
    )
    result["fixed_initial_true_anomaly_deg"] = (
        FIXED_INITIAL_TRUE_ANOMALY_DEG
    )

    return result


# ------------------------------------------------------------
# 5. HEADER
# ------------------------------------------------------------

total_expected_runs = (
    len(OUTAGE_DURATIONS_MIN)
    * NUMBER_OF_REPLICATES
)

print()
print("=" * 120)
print("AURORA — EXPERIMENT 017-G")
print("Mature-estimator outage-duration resilience sweep")
print("=" * 120)
print(
    f"Fixed initial true anomaly : "
    f"{FIXED_INITIAL_TRUE_ANOMALY_DEG:.1f} deg"
)
print(
    f"Fixed outage start : "
    f"{FIXED_OUTAGE_START_MIN:.1f} min"
)
print(
    f"Fixed estimator start : "
    f"{FIXED_ESTIMATOR_START_MIN:.1f} min"
)
print(
    f"Outage durations : "
    f"{OUTAGE_DURATIONS_MIN} min"
)
print(
    f"Replicates per duration : "
    f"{NUMBER_OF_REPLICATES}"
)
print(
    f"Total expected missions : "
    f"{total_expected_runs}"
)
print()


# ------------------------------------------------------------
# 6. DESIGN TABLE
# ------------------------------------------------------------

design_rows = []

for replicate_index in range(NUMBER_OF_REPLICATES):
    scenario_seed = BASE_SEED + replicate_index

    for duration_min in OUTAGE_DURATIONS_MIN:
        design_rows.append(
            {
                "outage_duration_min":
                    float(duration_min),
                "outage_start_min":
                    FIXED_OUTAGE_START_MIN,
                "outage_end_min":
                    FIXED_OUTAGE_START_MIN
                    + float(duration_min),
                "post_recovery_end_min":
                    FIXED_OUTAGE_START_MIN
                    + float(duration_min)
                    + POST_RECOVERY_WINDOW_MIN,
                "estimator_start_min":
                    FIXED_ESTIMATOR_START_MIN,
                "initial_true_anomaly_deg":
                    FIXED_INITIAL_TRUE_ANOMALY_DEG,
                "replicate_index":
                    int(replicate_index),
                "scenario_seed":
                    int(scenario_seed),
            }
        )

design = pd.DataFrame(design_rows)
design.to_csv(design_path, index=False)

design_valid = bool(
    len(design) == total_expected_runs
)


# ------------------------------------------------------------
# 7. CHECKPOINT / RESUME
# ------------------------------------------------------------

if results_path.exists():
    existing_results = pd.read_csv(results_path)

    required_keys = [
        "experiment_outage_duration_min",
        "replicate_index",
    ]

    if not all(
        column in existing_results.columns
        for column in required_keys
    ):
        raise RuntimeError(
            "Existing 017-G checkpoint has incompatible columns."
        )

    existing_results = (
        existing_results
        .sort_values(
            [
                "replicate_index",
                "experiment_outage_duration_min",
            ]
        )
        .drop_duplicates(
            subset=[
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
        float(row["outage_duration_min"]),
        int(row["replicate_index"]),
    )

    if key not in completed_keys:
        remaining_rows.append(row)

remaining_design = pd.DataFrame(remaining_rows)

print(
    f"Already completed : "
    f"{len(completed_keys)}"
)
print(
    f"Remaining : "
    f"{len(remaining_design)}"
)

if len(remaining_design) == 0:
    print(
        "Checkpoint complete — "
        "NO 017-G missions will be repeated."
    )

print()


# ------------------------------------------------------------
# 8. RUN MISSING MISSIONS
# ------------------------------------------------------------

new_rows = []

for local_index, (_, row) in enumerate(
    remaining_design.iterrows()
):
    duration_min = float(
        row["outage_duration_min"]
    )
    replicate_index = int(
        row["replicate_index"]
    )
    scenario_seed = int(
        row["scenario_seed"]
    )

    global_index = (
        len(completed_keys)
        + local_index
        + 1
    )

    print(
        f"[{global_index:02d}/{total_expected_runs}] "
        f"duration={duration_min:5.1f} min | "
        f"outage="
        f"{FIXED_OUTAGE_START_MIN:.1f}-"
        f"{FIXED_OUTAGE_START_MIN + duration_min:.1f} min | "
        f"rep={replicate_index:02d} | "
        f"seed={scenario_seed}"
    )

    result = run_duration_scenario(
        outage_duration_min=
            duration_min,
        replicate_index=
            replicate_index,
        scenario_seed=
            scenario_seed,
    )

    new_rows.append(result)

    if len(existing_results) > 0:
        checkpoint = pd.concat(
            [
                existing_results,
                pd.DataFrame(new_rows),
            ],
            ignore_index=True,
        )
    else:
        checkpoint = pd.DataFrame(new_rows)

    checkpoint = (
        checkpoint
        .sort_values(
            [
                "replicate_index",
                "experiment_outage_duration_min",
            ]
        )
        .drop_duplicates(
            subset=[
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
        f"{result['post_recovery_rmse_m']:.3f} m | "
        f"NIS="
        f"{result['navigation_nis_mean']:.3f}"
    )


# ------------------------------------------------------------
# 9. FINAL RESULTS
# ------------------------------------------------------------

results = pd.read_csv(results_path)

results = (
    results
    .sort_values(
        [
            "replicate_index",
            "experiment_outage_duration_min",
        ]
    )
    .drop_duplicates(
        subset=[
            "experiment_outage_duration_min",
            "replicate_index",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 10. SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        "experiment_outage_duration_min",
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
        sigma_start_std_m=(
            "outage_start_sigma_m",
            "std",
        ),
        outage_start_error_mean_m=(
            "outage_start_error_m",
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
        sigma_end_std_m=(
            "outage_end_sigma_m",
            "std",
        ),
        post_recovery_rmse_mean_m=(
            "post_recovery_rmse_m",
            "mean",
        ),
        post_recovery_rmse_std_m=(
            "post_recovery_rmse_m",
            "std",
        ),
        pdop_mean=(
            "pre_outage_pdop",
            "mean",
        ),
        visible_mean=(
            "natural_visible_satellites_mean",
            "mean",
        ),
        navigation_nis_mean=(
            "navigation_nis_mean",
            "mean",
        ),
        attitude_outage_rmse_mean_deg=(
            "attitude_outage_rmse_deg",
            "mean",
        ),
    )
)

summary.to_csv(
    summary_path,
    index=False,
)


# ------------------------------------------------------------
# 11. REPEATED-MEASURES TESTS
# ------------------------------------------------------------

PRIMARY_METRICS = [
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
    "post_recovery_rmse_m",
]

friedman_rows = []

for metric in PRIMARY_METRICS:
    pivot = (
        results.pivot(
            index="replicate_index",
            columns="experiment_outage_duration_min",
            values=metric,
        )
        .reindex(
            columns=OUTAGE_DURATIONS_MIN
        )
    )

    complete = pivot.dropna(axis=0)

    if len(complete) != NUMBER_OF_REPLICATES:
        raise RuntimeError(
            f"Incomplete repeated-measures matrix for {metric}."
        )

    samples = [
        complete[duration].to_numpy(dtype=float)
        for duration in OUTAGE_DURATIONS_MIN
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
                len(OUTAGE_DURATIONS_MIN)
                - 1
            )
        )
    )

    means = np.array(
        [
            np.mean(sample)
            for sample in samples
        ],
        dtype=float,
    )

    mean_range = float(
        np.max(means)
        - np.min(means)
    )

    overall_mean = float(
        np.mean(means)
    )

    friedman_rows.append(
        {
            "metric":
                metric,
            "friedman_statistic":
                float(statistic),
            "p_value":
                float(p_value),
            "kendalls_w":
                kendalls_w,
            "duration_mean_range":
                mean_range,
            "relative_duration_range_percent":
                float(
                    100.0
                    * mean_range
                    / max(
                        abs(overall_mean),
                        1.0e-12,
                    )
                ),
            "significant_0p05":
                bool(
                    p_value < 0.05
                ),
        }
    )

friedman_table = pd.DataFrame(
    friedman_rows
)

friedman_table.to_csv(
    statistics_path,
    index=False,
)


# ------------------------------------------------------------
# 12. SPEARMAN TREND TESTS
# ------------------------------------------------------------

correlation_pairs = [
    (
        "experiment_outage_duration_min",
        "outage_rmse_m",
    ),
    (
        "experiment_outage_duration_min",
        "outage_end_error_m",
    ),
    (
        "experiment_outage_duration_min",
        "outage_end_sigma_m",
    ),
    (
        "experiment_outage_duration_min",
        "post_recovery_rmse_m",
    ),
]

correlation_rows = []

for predictor, outcome in correlation_pairs:
    x = results[predictor].to_numpy(dtype=float)
    y = results[outcome].to_numpy(dtype=float)

    finite = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    rho, p_value = spearmanr(
        x[finite],
        y[finite],
    )

    correlation_rows.append(
        {
            "predictor":
                predictor,
            "outcome":
                outcome,
            "n":
                int(np.sum(finite)),
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
# 13. VALIDATION
# ------------------------------------------------------------

all_runs_complete = bool(
    len(results) == total_expected_runs
)

unique_keys = bool(
    results[
        [
            "experiment_outage_duration_min",
            "replicate_index",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    ==
    total_expected_runs
)

all_durations_present = bool(
    set(
        np.round(
            results[
                "experiment_outage_duration_min"
            ].astype(float),
            8,
        )
    )
    ==
    set(
        np.round(
            np.asarray(
                OUTAGE_DURATIONS_MIN,
                dtype=float,
            ),
            8,
        )
    )
)

all_replicates_per_duration = bool(
    np.all(
        results.groupby(
            "experiment_outage_duration_min"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)

fixed_outage_start_valid = bool(
    np.allclose(
        results[
            "experiment_outage_start_min"
        ].to_numpy(dtype=float),
        FIXED_OUTAGE_START_MIN,
    )
)

fixed_estimator_start_valid = bool(
    np.allclose(
        results[
            "experiment_estimator_start_min"
        ].to_numpy(dtype=float),
        FIXED_ESTIMATOR_START_MIN,
    )
)

fixed_phase_valid = bool(
    np.allclose(
        results[
            "fixed_initial_true_anomaly_deg"
        ].to_numpy(dtype=float),
        FIXED_INITIAL_TRUE_ANOMALY_DEG,
    )
)

core_metrics = [
    "outage_start_sigma_m",
    "outage_start_error_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
    "navigation_nis_mean",
    "attitude_outage_rmse_deg",
]

core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_metrics
            ].to_numpy(dtype=float)
        )
    )
)

recovery_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                "post_recovery_rmse_m"
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

# Because all outages begin at the same time and the estimator
# always starts at t=0, outage-start covariance should be
# essentially invariant across duration conditions for a
# given replicate.
sigma_pivot = results.pivot(
    index="replicate_index",
    columns="experiment_outage_duration_min",
    values="outage_start_sigma_m",
)

per_replicate_sigma_ranges = (
    sigma_pivot.max(axis=1)
    -
    sigma_pivot.min(axis=1)
)

start_sigma_invariant = bool(
    np.max(
        per_replicate_sigma_ranges.to_numpy(
            dtype=float
        )
    )
    <
    1.0e-9
)

validation_017g = bool(
    design_valid
    and
    all_runs_complete
    and
    unique_keys
    and
    all_durations_present
    and
    all_replicates_per_duration
    and
    fixed_outage_start_valid
    and
    fixed_estimator_start_valid
    and
    fixed_phase_valid
    and
    core_metrics_finite
    and
    recovery_metrics_finite
    and
    time_sync_valid
    and
    start_sigma_invariant
)

validation_table = pd.DataFrame(
    [
        {
            "design_valid":
                design_valid,
            "all_runs_complete":
                all_runs_complete,
            "unique_keys":
                unique_keys,
            "all_durations_present":
                all_durations_present,
            "all_replicates_per_duration":
                all_replicates_per_duration,
            "fixed_outage_start_valid":
                fixed_outage_start_valid,
            "fixed_estimator_start_valid":
                fixed_estimator_start_valid,
            "fixed_phase_valid":
                fixed_phase_valid,
            "core_metrics_finite":
                core_metrics_finite,
            "recovery_metrics_finite":
                recovery_metrics_finite,
            "time_sync_valid":
                time_sync_valid,
            "start_sigma_invariant":
                start_sigma_invariant,
            "global_validation":
                validation_017g,
        }
    ]
)

validation_table.to_csv(
    validation_path,
    index=False,
)


# ------------------------------------------------------------
# 14. PRINT
# ------------------------------------------------------------

print()
print("=" * 120)
print("AURORA — EXPERIMENT 017-G RESULTS")
print("Mature-estimator outage-duration resilience sweep")
print("=" * 120)
print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)
print()

print("----- DURATION SUMMARY -----")
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

print("----- REPEATED-MEASURES DURATION TEST -----")
print(
    friedman_table.to_string(
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
            "duration_mean_range":
                lambda value:
                    f"{value:.4f}",
            "relative_duration_range_percent":
                lambda value:
                    f"{value:.2f}",
        },
    )
)
print()

print("----- DURATION / ROBUSTNESS CORRELATIONS -----")
print(
    correlation_table.to_string(
        index=False,
        formatters={
            "spearman_rho":
                lambda value:
                    f"{value:.4f}",
            "p_value":
                lambda value:
                    f"{value:.6f}",
        },
    )
)
print()

worst_rmse_row = summary.loc[
    summary[
        "outage_rmse_mean_m"
    ].idxmax()
]

worst_end_row = summary.loc[
    summary[
        "outage_end_error_mean_m"
    ].idxmax()
]

best_rmse_row = summary.loc[
    summary[
        "outage_rmse_mean_m"
    ].idxmin()
]

print("----- RESILIENCE ENVELOPE ANCHORS -----")
print(
    f"Best mean outage RMSE : "
    f"{best_rmse_row['experiment_outage_duration_min']:.1f} min "
    f"-> {best_rmse_row['outage_rmse_mean_m']:.4f} m"
)
print(
    f"Worst mean outage RMSE : "
    f"{worst_rmse_row['experiment_outage_duration_min']:.1f} min "
    f"-> {worst_rmse_row['outage_rmse_mean_m']:.4f} m"
)
print(
    f"Worst mean end error : "
    f"{worst_end_row['experiment_outage_duration_min']:.1f} min "
    f"-> {worst_end_row['outage_end_error_mean_m']:.4f} m"
)
print()

print("----- VALIDATION -----")
print(
    f"All {total_expected_runs} missions complete : "
    f"{all_runs_complete}"
)
print(
    f"All duration/replicate keys unique : "
    f"{unique_keys}"
)
print(
    f"8 replicates per duration : "
    f"{all_replicates_per_duration}"
)
print(
    f"Outage start fixed at 60 min : "
    f"{fixed_outage_start_valid}"
)
print(
    f"Estimator start fixed at 0 min : "
    f"{fixed_estimator_start_valid}"
)
print(
    f"Initial orbital phase fixed : "
    f"{fixed_phase_valid}"
)
print(
    f"All primary metrics finite : "
    f"{core_metrics_finite}"
)
print(
    f"15-min post-recovery metric finite : "
    f"{recovery_metrics_finite}"
)
print(
    f"Outage-start sigma invariant across durations : "
    f"{start_sigma_invariant}"
)
print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)
print()
print(
    f"VALIDATION GLOBALE 017-G : "
    f"{validation_017g}"
)
print("=" * 120)


# ------------------------------------------------------------
# 15. FIGURES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.errorbar(
    summary[
        "experiment_outage_duration_min"
    ],
    summary[
        "outage_rmse_mean_m"
    ],
    yerr=
        summary[
            "outage_rmse_std_m"
        ],
    marker="o",
)
plt.xlabel("Dual-outage duration [min]")
plt.ylabel("Outage RMSE [m]")
plt.title(
    "AURORA — 017-G outage duration vs navigation RMSE"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017g_duration_vs_outage_rmse.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 6))
plt.errorbar(
    summary[
        "experiment_outage_duration_min"
    ],
    summary[
        "outage_end_error_mean_m"
    ],
    yerr=
        summary[
            "outage_end_error_std_m"
        ],
    marker="o",
)
plt.xlabel("Dual-outage duration [min]")
plt.ylabel("End-of-outage position error [m]")
plt.title(
    "AURORA — 017-G outage duration vs terminal error"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017g_duration_vs_end_error.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(
    summary[
        "experiment_outage_duration_min"
    ],
    summary[
        "sigma_end_mean_m"
    ],
    marker="o",
)
plt.xlabel("Dual-outage duration [min]")
plt.ylabel("Position sigma at outage end [m]")
plt.title(
    "AURORA — 017-G covariance growth with outage duration"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017g_duration_vs_end_sigma.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 6))
for replicate_index in sorted(
    results[
        "replicate_index"
    ].astype(int).unique()
):
    block = (
        results[
            results[
                "replicate_index"
            ]
            ==
            replicate_index
        ]
        .sort_values(
            "experiment_outage_duration_min"
        )
    )

    plt.plot(
        block[
            "experiment_outage_duration_min"
        ],
        block[
            "outage_rmse_m"
        ],
        marker="o",
    )

plt.xlabel("Dual-outage duration [min]")
plt.ylabel("Outage RMSE [m]")
plt.title(
    "AURORA — 017-G paired duration response"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017g_paired_duration_response.png",
    dpi=200,
)
plt.close()
