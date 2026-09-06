from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare
)

from src.validation.outage_phase_sweep import (
    PHASES_DEG,
    DUAL_OUTAGE_START_MIN,
    DUAL_OUTAGE_END_MIN,
    run_dual_outage_phase_scenario
)


# ============================================================
# AURORA
# Experience 017-C
#
# DUAL-OUTAGE ORBITAL-PHASE SWEEP
#
#
# We keep:
#
#       altitude = 550 km
#       i        = 97.6 deg
#
#
# Initial orbital phase:
#
#       0,45,...315 deg
#
#
# Dual GNSS + star tracker outage:
#
#       60 - 75 min
#
#
# Every phase uses:
#
#       8 stochastic seed blocks
#
#
# Total:
#
#       8 phases * 8 replicates
#       =
#       64 full missions
#
#
# Main statistical test:
#
#       Friedman repeated-measures test
#
# because each replicate seed is reused across orbital
# phases.
#
#
# NOTE:
#
# Reusing a seed creates a stochastic block, but because
# visible satellite counts/order can differ by phase,
# GNSS random-number consumption is not guaranteed to be
# exactly identical epoch-by-epoch.
# ============================================================


# ------------------------------------------------------------
# 1. CAMPAIGN
# ------------------------------------------------------------

NUMBER_OF_REPLICATES = (
    8
)


BASE_SEED = (
    170_300
)


# ------------------------------------------------------------
# 2. OUTPUT
# ------------------------------------------------------------

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


data_directory = (
    Path("data")
    /
    "phase17"
)


table_directory.mkdir(
    parents=True,
    exist_ok=True
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


data_directory.mkdir(
    parents=True,
    exist_ok=True
)


design_path = (
    data_directory
    /
    "dual_outage_phase_design_017c.csv"
)


results_path = (
    table_directory
    /
    "phase17_017c_dual_outage_phase_sweep.csv"
)


summary_path = (
    table_directory
    /
    "phase17_017c_phase_summary.csv"
)


statistics_path = (
    table_directory
    /
    "phase17_017c_friedman_statistics.csv"
)


# ------------------------------------------------------------
# 3. FIXED DESIGN
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


    for phase_deg in PHASES_DEG:

        design_rows.append(
            {
                "phase_name":
                    (
                        f"PHASE_"
                        f"{int(round(phase_deg)):03d}"
                    ),

                "initial_true_anomaly_deg":
                    float(
                        phase_deg
                    ),

                "replicate_index":
                    int(
                        replicate_index
                    ),

                "scenario_seed":
                    int(
                        scenario_seed
                    )
            }
        )


design = pd.DataFrame(
    design_rows
)


design.to_csv(
    design_path,
    index=False
)


total_expected_runs = (
    len(
        PHASES_DEG
    )
    *
    NUMBER_OF_REPLICATES
)


design_valid = (
    len(
        design
    )
    ==
    total_expected_runs
)


# ------------------------------------------------------------
# 4. CHECKPOINT
# ------------------------------------------------------------

if results_path.exists():

    existing_results = pd.read_csv(
        results_path
    )


    completed_keys = set(
        zip(
            existing_results[
                "initial_true_anomaly_deg"
            ].astype(
                float
            ),

            existing_results[
                "replicate_index"
            ].astype(
                int
            )
        )
    )


else:

    existing_results = pd.DataFrame()

    completed_keys = set()


remaining_design = design[
    [
        (
            float(
                row[
                    "initial_true_anomaly_deg"
                ]
            ),
            int(
                row[
                    "replicate_index"
                ]
            )
        )
        not in completed_keys

        for _, row in design.iterrows()
    ]
].copy()


# ------------------------------------------------------------
# 5. HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-C"
)


print(
    "Dual-outage orbital-phase sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Initial orbital phases : "
    f"{PHASES_DEG} deg"
)


print(
    f"Replicates per phase : "
    f"{NUMBER_OF_REPLICATES}"
)


print(
    f"Dual outage : "
    f"{DUAL_OUTAGE_START_MIN:.1f} - "
    f"{DUAL_OUTAGE_END_MIN:.1f} min"
)


print(
    f"Total full missions : "
    f"{total_expected_runs}"
)


print(
    f"Already completed : "
    f"{len(completed_keys)}"
)


print(
    f"Remaining : "
    f"{len(remaining_design)}"
)


print()


# ------------------------------------------------------------
# 6. RUN
# ------------------------------------------------------------

new_rows = []


for local_index, (_, row) in enumerate(
    remaining_design.iterrows()
):

    phase_deg = float(
        row[
            "initial_true_anomaly_deg"
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
        len(
            completed_keys
        )
        +
        local_index
        +
        1
    )


    print(
        f"[{global_index:02d}/{total_expected_runs}] "
        f"nu0={phase_deg:6.1f} deg | "
        f"rep={replicate_index:02d} | "
        f"seed={scenario_seed}"
    )


    result = (
        run_dual_outage_phase_scenario(
            initial_true_anomaly_deg=
                phase_deg,

            replicate_index=
                replicate_index,

            scenario_seed=
                scenario_seed
        )
    )


    new_rows.append(
        result
    )


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
            [
                "replicate_index",
                "initial_true_anomaly_deg"
            ]
        )
        .drop_duplicates(
            subset=[
                "initial_true_anomaly_deg",
                "replicate_index"
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
        f"    PDOPpre="
        f"{result['pre_outage_pdop']:.3f} | "
        f"RMSE outage="
        f"{result['outage_rmse_m']:.3f} m | "
        f"end="
        f"{result['outage_end_error_m']:.3f} m | "
        f"sigma_end="
        f"{result['outage_end_sigma_m']:.3f} m | "
        f"att="
        f"{result['attitude_outage_rmse_deg']:.4f} deg"
    )


# ------------------------------------------------------------
# 7. RELOAD
# ------------------------------------------------------------

results = pd.read_csv(
    results_path
)


results = (
    results.sort_values(
        [
            "replicate_index",
            "initial_true_anomaly_deg"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 8. CORE VALIDATION
# ------------------------------------------------------------

all_runs_complete = (
    len(
        results
    )
    ==
    total_expected_runs
)


unique_keys = (
    results[
        [
            "initial_true_anomaly_deg",
            "replicate_index"
        ]
    ]
    .drop_duplicates()
    .shape[
        0
    ]
    ==
    total_expected_runs
)


all_phases_complete = bool(
    np.all(
        results.groupby(
            "initial_true_anomaly_deg"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)


all_replicates_cover_all_phases = bool(
    np.all(
        results.groupby(
            "replicate_index"
        )[
            "initial_true_anomaly_deg"
        ]
        .nunique()
        ==
        len(
            PHASES_DEG
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


core_columns = [
    "pre_outage_pdop",

    "pre_outage_rmse_m",

    "mission_rmse_m",

    "outage_rmse_m",

    "outage_start_error_m",

    "outage_end_error_m",

    "outage_start_sigma_m",

    "outage_end_sigma_m",

    "post_recovery_rmse_m",

    "final_error_m",

    "navigation_nis_mean",

    "navigation_nis_coverage_percent",

    "attitude_outage_rmse_deg",

    "attitude_end_outage_deg",

    "healthy_false_alarm_rate"
]


all_core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_columns
            ].to_numpy(
                dtype=float
            )
        )
    )
)


outage_duration_constant = bool(
    np.allclose(
        results[
            "dual_outage_duration_min"
        ].to_numpy(
            dtype=float
        ),
        15.0
    )
)


# ------------------------------------------------------------
# 9. PHASE SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        "initial_true_anomaly_deg",
        as_index=False
    )
    .agg(
        runs=(
            "replicate_index",
            "size"
        ),

        visible_mean=(
            "natural_visible_satellites_mean",
            "mean"
        ),

        pre_outage_pdop_mean=(
            "pre_outage_pdop",
            "mean"
        ),

        pre_outage_pdop_std=(
            "pre_outage_pdop",
            "std"
        ),

        pre_outage_update_age_mean_s=(
            "pre_outage_update_age_s",
            "mean"
        ),

        pre_outage_rmse_mean=(
            "pre_outage_rmse_m",
            "mean"
        ),

        outage_rmse_mean=(
            "outage_rmse_m",
            "mean"
        ),

        outage_rmse_std=(
            "outage_rmse_m",
            "std"
        ),

        outage_end_error_mean=(
            "outage_end_error_m",
            "mean"
        ),

        outage_end_error_std=(
            "outage_end_error_m",
            "std"
        ),

        outage_end_sigma_mean=(
            "outage_end_sigma_m",
            "mean"
        ),

        post_recovery_rmse_mean=(
            "post_recovery_rmse_m",
            "mean"
        ),

        post_recovery_rmse_std=(
            "post_recovery_rmse_m",
            "std"
        ),

        attitude_outage_rmse_mean_deg=(
            "attitude_outage_rmse_deg",
            "mean"
        ),

        navigation_nis_mean=(
            "navigation_nis_mean",
            "mean"
        ),

        healthy_false_alarm_mean=(
            "healthy_false_alarm_rate",
            "mean"
        )
    )
)


summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 10. FRIEDMAN TEST
#
# Repeated-measures non-parametric test.
#
# H0:
#
#       all orbital phases have the same distribution
#
# across the replicate blocks.
# ------------------------------------------------------------

FRIEDMAN_METRICS = [
    "outage_rmse_m",
    "outage_end_error_m",
    "post_recovery_rmse_m",
    "pre_outage_pdop"
]


friedman_rows = []


for metric in FRIEDMAN_METRICS:

    pivot = (
        results.pivot(
            index=
                "replicate_index",

            columns=
                "initial_true_anomaly_deg",

            values=
                metric
        )
        .reindex(
            columns=
                PHASES_DEG
        )
    )


    if pivot.isna().any().any():

        raise RuntimeError(
            f"Incomplete repeated-measures matrix: "
            f"{metric}"
        )


    phase_samples = [
        pivot[
            phase_deg
        ].to_numpy(
            dtype=float
        )

        for phase_deg in PHASES_DEG
    ]


    statistic, p_value = (
        friedmanchisquare(
            *phase_samples
        )
    )


    number_of_blocks = (
        NUMBER_OF_REPLICATES
    )


    number_of_phases = len(
        PHASES_DEG
    )


    kendalls_w = (
        statistic
        /
        (
            number_of_blocks
            *
            (
                number_of_phases
                -
                1
            )
        )
    )


    phase_means = np.array(
        [
            np.mean(
                sample
            )

            for sample in phase_samples
        ],
        dtype=float
    )


    mean_range = float(
        np.max(
            phase_means
        )
        -
        np.min(
            phase_means
        )
    )


    relative_range_percent = float(
        100.0
        *
        mean_range
        /
        max(
            np.mean(
                phase_means
            ),
            1.0e-12
        )
    )


    friedman_rows.append(
        {
            "metric":
                metric,

            "friedman_statistic":
                float(
                    statistic
                ),

            "p_value":
                float(
                    p_value
                ),

            "kendalls_w":
                float(
                    kendalls_w
                ),

            "phase_mean_range":
                mean_range,

            "relative_phase_range_percent":
                relative_range_percent,

            "significant_0p05":
                bool(
                    p_value
                    <
                    0.05
                )
        }
    )


friedman_table = pd.DataFrame(
    friedman_rows
)


friedman_table.to_csv(
    statistics_path,
    index=False
)


# ------------------------------------------------------------
# 11. BEST / WORST PHASES
# ------------------------------------------------------------

best_outage_index = (
    summary[
        "outage_rmse_mean"
    ].idxmin()
)


worst_outage_index = (
    summary[
        "outage_rmse_mean"
    ].idxmax()
)


best_phase = (
    summary.loc[
        best_outage_index
    ]
)


worst_phase = (
    summary.loc[
        worst_outage_index
    ]
)


outage_metric_row = (
    friedman_table[
        friedman_table[
            "metric"
        ]
        ==
        "outage_rmse_m"
    ]
    .iloc[
        0
    ]
)


end_error_metric_row = (
    friedman_table[
        friedman_table[
            "metric"
        ]
        ==
        "outage_end_error_m"
    ]
    .iloc[
        0
    ]
)


# ------------------------------------------------------------
# 12. SCIENTIFIC FLAGS
# ------------------------------------------------------------

orbital_phase_changes_outage_rmse = bool(
    outage_metric_row[
        "p_value"
    ]
    <
    0.05
)


orbital_phase_changes_end_error = bool(
    end_error_metric_row[
        "p_value"
    ]
    <
    0.05
)


outage_phase_effect_large = bool(
    outage_metric_row[
        "kendalls_w"
    ]
    >=
    0.30
)


all_nis_reasonable = bool(
    np.all(
        (
            summary[
                "navigation_nis_mean"
            ]
            >=
            1.5
        )
        &
        (
            summary[
                "navigation_nis_mean"
            ]
            <=
            4.5
        )
    )
)


# ------------------------------------------------------------
# 13. VALIDATION
# ------------------------------------------------------------

validation_017c = (
    design_valid
    and
    all_runs_complete
    and
    unique_keys
    and
    all_phases_complete
    and
    all_replicates_cover_all_phases
    and
    time_sync_valid
    and
    all_core_metrics_finite
    and
    outage_duration_constant
)


# ------------------------------------------------------------
# 14. PRINT
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-C RESULTS"
)


print(
    "Dual-outage orbital-phase sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)


print(
    f"Orbital phases : "
    f"{len(PHASES_DEG)}"
)


print(
    f"Replicates per phase : "
    f"{NUMBER_OF_REPLICATES}"
)


print(
    f"Fixed dual outage : "
    f"{DUAL_OUTAGE_START_MIN:.1f} - "
    f"{DUAL_OUTAGE_END_MIN:.1f} min"
)


print()


# ------------------------------------------------------------
# 15. SUMMARY
# ------------------------------------------------------------

print(
    "----- PHASE SUMMARY -----"
)


print(
    summary.to_string(
        index=False,
        formatters={
            column:
                lambda value:
                    f"{value:.4f}"

            for column in summary.columns

            if column
            not in [
                "runs"
            ]
        }
    )
)


print()


# ------------------------------------------------------------
# 16. STATISTICAL TEST
# ------------------------------------------------------------

print(
    "----- REPEATED-MEASURES PHASE TEST -----"
)


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

            "phase_mean_range":
                lambda value:
                    f"{value:.4f}",

            "relative_phase_range_percent":
                lambda value:
                    f"{value:.2f}"
        }
    )
)


print()


# ------------------------------------------------------------
# 17. BEST / WORST
# ------------------------------------------------------------

print(
    "----- BEST / WORST OUTAGE PHASE -----"
)


print(
    f"Best phase by outage RMSE : "
    f"nu0={best_phase['initial_true_anomaly_deg']:.1f} deg | "
    f"RMSE={best_phase['outage_rmse_mean']:.3f} m"
)


print(
    f"Worst phase by outage RMSE : "
    f"nu0={worst_phase['initial_true_anomaly_deg']:.1f} deg | "
    f"RMSE={worst_phase['outage_rmse_mean']:.3f} m"
)


print(
    f"Mean phase range : "
    f"{outage_metric_row['phase_mean_range']:.3f} m "
    f"("
    f"{outage_metric_row['relative_phase_range_percent']:.1f}%"
    f")"
)


print()


# ------------------------------------------------------------
# 18. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Orbital phase significantly changes outage RMSE : "
    f"{orbital_phase_changes_outage_rmse}"
)


print(
    f"Orbital phase significantly changes end-of-outage error : "
    f"{orbital_phase_changes_end_error}"
)


print(
    f"Outage phase effect Kendall W >= 0.30 : "
    f"{outage_phase_effect_large}"
)


print(
    f"Protected NIS remains in [1.5, 4.5] for every phase : "
    f"{all_nis_reasonable}"
)


print()


# ------------------------------------------------------------
# 19. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"Fixed design valid : "
    f"{design_valid}"
)


print(
    f"All 64 missions complete : "
    f"{all_runs_complete}"
)


print(
    f"All phase/replicate keys unique : "
    f"{unique_keys}"
)


print(
    f"8 replicates available for every phase : "
    f"{all_phases_complete}"
)


print(
    f"Every replicate covers all 8 phases : "
    f"{all_replicates_cover_all_phases}"
)


print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)


print(
    f"All core metrics finite : "
    f"{all_core_metrics_finite}"
)


print(
    f"Dual-outage duration fixed at 15 min : "
    f"{outage_duration_constant}"
)


print()


print(
    f"VALIDATION GLOBALE 017-C : "
    f"{validation_017c}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 20. FIGURE — OUTAGE RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "initial_true_anomaly_deg"
    ],

    summary[
        "outage_rmse_mean"
    ],

    yerr=
        summary[
            "outage_rmse_std"
        ],

    marker="o"
)


plt.xlabel(
    "Initial true anomaly [deg]"
)


plt.ylabel(
    "Protected dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-C outage resilience vs orbital phase"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017c_outage_rmse_vs_phase.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 21. FIGURE — END OF OUTAGE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "initial_true_anomaly_deg"
    ],

    summary[
        "outage_end_error_mean"
    ],

    yerr=
        summary[
            "outage_end_error_std"
        ],

    marker="o"
)


plt.xlabel(
    "Initial true anomaly [deg]"
)


plt.ylabel(
    "Position error at end of dual outage [m]"
)


plt.title(
    "AURORA — 017-C end-of-outage error vs orbital phase"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017c_end_error_vs_phase.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 22. FIGURE — PRE-OUTAGE PDOP
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "initial_true_anomaly_deg"
    ],

    summary[
        "pre_outage_pdop_mean"
    ],

    yerr=
        summary[
            "pre_outage_pdop_std"
        ],

    marker="o"
)


plt.xlabel(
    "Initial true anomaly [deg]"
)


plt.ylabel(
    "PDOP immediately before outage"
)


plt.title(
    "AURORA — 017-C GNSS geometry before dual outage"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017c_pre_outage_pdop_vs_phase.png",

    dpi=200
)


plt.show()