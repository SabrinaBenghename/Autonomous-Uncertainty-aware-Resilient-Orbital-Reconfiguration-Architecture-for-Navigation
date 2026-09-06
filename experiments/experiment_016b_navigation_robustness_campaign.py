from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.validation.full_mission_resilience import (
    run_full_mission_robustness_scenario
)


# ============================================================
# AURORA
# Experience 016-B
#
# FULL NAVIGATION ROBUSTNESS CAMPAIGN
#
#
# Input:
#
#       64 fixed scenarios from 016-A
#
#
# Compare:
#
#       PERMISSIVE
#
# versus
#
#       AURORA PROTECTED
#
#
# IMPORTANT:
#
# Results are checkpointed after EVERY scenario.
#
# If execution stops:
#
#       run the exact same command again
#
# and completed scenario IDs are automatically skipped.
#
#
# AI is deliberately excluded.
# Phase 16-C handles AI robustness separately.
# ============================================================


# ------------------------------------------------------------
# 1. INPUT / OUTPUT
# ------------------------------------------------------------

campaign_path = (
    Path("data")
    /
    "phase16"
    /
    "robustness_campaign_016a.csv"
)


results_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016b_navigation_robustness.csv"
)


summary_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016b_navigation_summary.csv"
)


worst_cases_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016b_worst_cases.csv"
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
        "016-A campaign not found: "
        f"{campaign_path}"
    )


# ------------------------------------------------------------
# 2. LOAD IMMUTABLE CAMPAIGN
# ------------------------------------------------------------

campaign = pd.read_csv(
    campaign_path
)


number_of_scenarios = len(
    campaign
)


if number_of_scenarios != 64:

    raise RuntimeError(
        "016-B expects exactly 64 scenarios "
        "from validated 016-A."
    )


# ------------------------------------------------------------
# 3. LOAD CHECKPOINT IF PRESENT
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


print(
    "\n"
    "========================================================================================================================"
)

print(
    "AURORA — Experience 016-B"
)

print(
    "Full navigation robustness campaign"
)

print(
    "========================================================================================================================"
)

print(
    f"Campaign scenarios : "
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

print()

print(
    "NOTE:"
)

print(
    "Each completed scenario is checkpointed immediately."
)

print(
    "If execution is interrupted, rerun the same command."
)

print()


# ------------------------------------------------------------
# 4. RUN REMAINING SCENARIOS
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

    global_position = (
        len(
            completed_ids
        )
        +
        local_index
        +
        1
    )

    print(
        f"[{global_position:02d}/{number_of_scenarios}] "
        f"{scenario_id} | "
        f"sigma_rho={scenario['pseudorange_noise_std_m']:.2f} m | "
        f"fault={scenario['fault_magnitude_m']:.1f} m | "
        f"duration={scenario['fault_duration_min']:.1f} min"
    )

    result = (
        run_full_mission_robustness_scenario(
            scenario
        )
    )

    merged_result = {
        **scenario.to_dict(),
        **result
    }

    new_rows.append(
        merged_result
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
        f"    FDIR low P_D = "
        f"{result['low_detection_rate']:.1f}% | "
        f"RMSE low P/R = "
        f"{result['permissive_low_redundancy_rmse']:.3f} / "
        f"{result['protected_low_redundancy_rmse']:.3f} m | "
        f"dual P/R = "
        f"{result['permissive_dual_outage_rmse']:.3f} / "
        f"{result['protected_dual_outage_rmse']:.3f} m | "
        f"NIS protected = "
        f"{result['protected_nis_mean']:.3f}"
    )


# ------------------------------------------------------------
# 5. RELOAD COMPLETE RESULTS
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
# 6. CORE CAMPAIGN VALIDATION
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


scenario_order_complete = bool(
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


# ------------------------------------------------------------
# 7. CORE FINITE METRICS
# ------------------------------------------------------------

core_metric_columns = [
    "strong_detection_rate",
    "strong_isolation_rate",
    "low_detection_rate",
    "false_alarm_rate",

    "attitude_rmse_deg",
    "attitude_nis_mean",

    "permissive_mission_rmse",
    "protected_mission_rmse",

    "permissive_low_redundancy_rmse",
    "protected_low_redundancy_rmse",

    "permissive_dual_outage_rmse",
    "protected_dual_outage_rmse",

    "permissive_nis_mean",
    "protected_nis_mean",

    "protected_low_update_rate"
]


core_metric_matrix = results[
    core_metric_columns
].to_numpy(
    dtype=float
)


all_core_metrics_finite = bool(
    np.all(
        np.isfinite(
            core_metric_matrix
        )
    )
)


# ------------------------------------------------------------
# 8. ARCHITECTURAL INTEGRITY PROPERTY
#
# Protected policy must NEVER use low-redundancy GNSS
# updates because exactly 5 satellites are deliberately
# supplied during that event.
# ------------------------------------------------------------

protected_blocks_low_redundancy = bool(
    np.all(
        np.abs(
            results[
                "protected_low_update_rate"
            ].to_numpy(
                dtype=float
            )
        )
        <
        1.0e-12
    )
)


runs_with_missed_faults = (
    results[
        "low_fault_missed_epochs"
    ].to_numpy(
        dtype=int
    )
    >
    0
)


fraction_runs_with_missed_faults = (
    100.0
    *
    np.mean(
        runs_with_missed_faults
    )
)


if np.any(
    runs_with_missed_faults
):

    protected_missed_update_values = (
        results.loc[
            runs_with_missed_faults,
            "protected_missed_update_rate"
        ].to_numpy(
            dtype=float
        )
    )

    permissive_missed_update_values = (
        results.loc[
            runs_with_missed_faults,
            "permissive_missed_update_rate"
        ].to_numpy(
            dtype=float
        )
    )

    protected_blocks_missed_faults = bool(
        np.all(
            np.abs(
                protected_missed_update_values
            )
            <
            1.0e-12
        )
    )

    mean_permissive_missed_update_rate = float(
        np.mean(
            permissive_missed_update_values
        )
    )

else:

    protected_blocks_missed_faults = (
        True
    )

    mean_permissive_missed_update_rate = (
        np.nan
    )


# ------------------------------------------------------------
# 9. DISTRIBUTION HELPERS
# ------------------------------------------------------------

def summarize_metric(
    metric_name,
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

    return {
        "metric":
            metric_name,

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90.0
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95.0
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            )
    }


summary_rows = []


summary_metric_columns = [
    "strong_detection_rate",
    "strong_isolation_rate",
    "low_detection_rate",
    "false_alarm_rate",

    "attitude_rmse_deg",
    "attitude_nis_mean",

    "permissive_mission_rmse",
    "protected_mission_rmse",

    "permissive_low_redundancy_rmse",
    "protected_low_redundancy_rmse",

    "permissive_post_low_rmse",
    "protected_post_low_rmse",

    "permissive_dual_outage_rmse",
    "protected_dual_outage_rmse",

    "permissive_post_dual_rmse",
    "protected_post_dual_rmse",

    "permissive_nis_mean",
    "protected_nis_mean",

    "permissive_nis_coverage",
    "protected_nis_coverage",

    "gain_mission_rmse_m",
    "gain_low_rmse_m",
    "gain_dual_rmse_m",

    "ratio_low_rmse",
    "ratio_dual_rmse"
]


for column in summary_metric_columns:

    summary_rows.append(
        summarize_metric(
            column,
            results[
                column
            ]
        )
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 10. PAIRED ROBUSTNESS RESULTS
# ------------------------------------------------------------

protected_better_mission = (
    results[
        "protected_mission_rmse"
    ]
    <
    results[
        "permissive_mission_rmse"
    ]
)


protected_better_low = (
    results[
        "protected_low_redundancy_rmse"
    ]
    <
    results[
        "permissive_low_redundancy_rmse"
    ]
)


protected_better_dual = (
    results[
        "protected_dual_outage_rmse"
    ]
    <
    results[
        "permissive_dual_outage_rmse"
    ]
)


fraction_protected_better_mission = (
    100.0
    *
    np.mean(
        protected_better_mission
    )
)


fraction_protected_better_low = (
    100.0
    *
    np.mean(
        protected_better_low
    )
)


fraction_protected_better_dual = (
    100.0
    *
    np.mean(
        protected_better_dual
    )
)


# ------------------------------------------------------------
# 11. NIS ROBUSTNESS
# ------------------------------------------------------------

protected_nis_values = (
    results[
        "protected_nis_mean"
    ].to_numpy(
        dtype=float
    )
)


permissive_nis_values = (
    results[
        "permissive_nis_mean"
    ].to_numpy(
        dtype=float
    )
)


protected_nis_coherent = (
    (
        protected_nis_values
        >=
        1.5
    )
    &
    (
        protected_nis_values
        <=
        4.5
    )
)


permissive_nis_coherent = (
    (
        permissive_nis_values
        >=
        1.5
    )
    &
    (
        permissive_nis_values
        <=
        4.5
    )
)


protected_nis_coherent_fraction = (
    100.0
    *
    np.mean(
        protected_nis_coherent
    )
)


permissive_nis_coherent_fraction = (
    100.0
    *
    np.mean(
        permissive_nis_coherent
    )
)


# ------------------------------------------------------------
# 12. FDIR ENSEMBLE
# ------------------------------------------------------------

mean_strong_detection = float(
    np.mean(
        results[
            "strong_detection_rate"
        ]
    )
)


mean_strong_isolation = float(
    np.mean(
        results[
            "strong_isolation_rate"
        ]
    )
)


mean_low_detection = float(
    np.mean(
        results[
            "low_detection_rate"
        ]
    )
)


mean_false_alarm = float(
    np.mean(
        results[
            "false_alarm_rate"
        ]
    )
)


# ------------------------------------------------------------
# 13. WORST CASES
# ------------------------------------------------------------

worst_case_rows = []


worst_definitions = {
    "protected_mission_rmse":
        "Worst protected mission RMSE",

    "protected_low_redundancy_rmse":
        "Worst protected low-redundancy RMSE",

    "protected_dual_outage_rmse":
        "Worst protected dual-outage RMSE",

    "protected_nis_mean":
        "Largest protected mean NIS",

    "permissive_mission_rmse":
        "Worst permissive mission RMSE"
}


for metric_column, description in worst_definitions.items():

    worst_index = (
        results[
            metric_column
        ].idxmax()
    )

    row = (
        results.loc[
            worst_index
        ].copy()
    )

    row[
        "worst_case_definition"
    ] = (
        description
    )

    row[
        "worst_case_metric"
    ] = (
        metric_column
    )

    worst_case_rows.append(
        row
    )


worst_cases = pd.DataFrame(
    worst_case_rows
)


worst_cases.to_csv(
    worst_cases_path,
    index=False
)


# ------------------------------------------------------------
# 14. SCIENTIFIC FLAGS
#
# These are FINDINGS, not requirements for experimental
# methodology validation.
# ------------------------------------------------------------

protected_low_better_majority = (
    fraction_protected_better_low
    >
    75.0
)


protected_dual_better_majority = (
    fraction_protected_better_dual
    >
    75.0
)


protected_mission_better_majority = (
    fraction_protected_better_mission
    >
    75.0
)


protected_nis_majority_coherent = (
    protected_nis_coherent_fraction
    >
    70.0
)


# ------------------------------------------------------------
# 15. 016-B VALIDATION
#
# Validation means:
#
#   - campaign executed correctly
#   - results are finite
#   - scenarios match 016-A
#   - protected integrity gate never consumes 5-sat data
#
# It does NOT force AURORA to outperform in every stress case.
# ------------------------------------------------------------

validation_016b = (
    all_scenarios_complete
    and
    scenario_order_complete
    and
    time_sync_valid
    and
    all_core_metrics_finite
    and
    protected_blocks_low_redundancy
    and
    protected_blocks_missed_faults
)


# ------------------------------------------------------------
# 16. PRINT
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)

print(
    "AURORA — Experience 016-B RESULTS"
)

print(
    "Full navigation robustness campaign"
)

print(
    "========================================================================================================================"
)

print(
    f"Scenarios completed : "
    f"{len(results)}/{number_of_scenarios}"
)

print(
    f"Runs with at least one missed low-redundancy fault epoch : "
    f"{fraction_runs_with_missed_faults:.2f} %"
)

print()


# ------------------------------------------------------------
# 17. FDIR
# ------------------------------------------------------------

print(
    "----- FDIR ROBUSTNESS -----"
)

print(
    f"Mean strong-fault detection : "
    f"{mean_strong_detection:.2f} %"
)

print(
    f"Mean strong-fault correct isolation : "
    f"{mean_strong_isolation:.2f} %"
)

print(
    f"Mean low-redundancy fault detection : "
    f"{mean_low_detection:.2f} %"
)

print(
    f"Mean false alarm rate : "
    f"{mean_false_alarm:.2f} %"
)

print()

print(
    f"Mean permissive update rate on MISSED low-red faults : "
    f"{mean_permissive_missed_update_rate:.2f} %"
)

print(
    f"Protected blocks all low-redundancy updates : "
    f"{protected_blocks_low_redundancy}"
)

print(
    f"Protected blocks all MISSED low-red faults : "
    f"{protected_blocks_missed_faults}"
)

print()


# ------------------------------------------------------------
# 18. PAIRED ARCHITECTURE PERFORMANCE
# ------------------------------------------------------------

print(
    "----- PAIRED ARCHITECTURE PERFORMANCE -----"
)

print(
    f"Protected better mission RMSE : "
    f"{fraction_protected_better_mission:.2f} % of scenarios"
)

print(
    f"Protected better low-redundancy RMSE : "
    f"{fraction_protected_better_low:.2f} % of scenarios"
)

print(
    f"Protected better dual-outage RMSE : "
    f"{fraction_protected_better_dual:.2f} % of scenarios"
)

print()


# ------------------------------------------------------------
# 19. DISTRIBUTION TABLE
# ------------------------------------------------------------

interesting_metrics = [
    "permissive_mission_rmse",
    "protected_mission_rmse",

    "permissive_low_redundancy_rmse",
    "protected_low_redundancy_rmse",

    "permissive_dual_outage_rmse",
    "protected_dual_outage_rmse",

    "permissive_nis_mean",
    "protected_nis_mean",

    "gain_mission_rmse_m",
    "gain_low_rmse_m",
    "gain_dual_rmse_m"
]


print(
    "----- ROBUSTNESS DISTRIBUTIONS -----"
)


display_summary = summary[
    summary[
        "metric"
    ].isin(
        interesting_metrics
    )
]


print(
    display_summary.to_string(
        index=False,
        formatters={
            "mean":
                lambda value:
                    f"{value:.3f}",

            "std":
                lambda value:
                    f"{value:.3f}",

            "median":
                lambda value:
                    f"{value:.3f}",

            "p90":
                lambda value:
                    f"{value:.3f}",

            "p95":
                lambda value:
                    f"{value:.3f}",

            "max":
                lambda value:
                    f"{value:.3f}"
        }
    )
)

print()


# ------------------------------------------------------------
# 20. NIS
# ------------------------------------------------------------

print(
    "----- CONSISTENCY ROBUSTNESS -----"
)

print(
    f"Protected NIS coherent [1.5, 4.5] : "
    f"{protected_nis_coherent_fraction:.2f} % of scenarios"
)

print(
    f"Permissive NIS coherent [1.5, 4.5] : "
    f"{permissive_nis_coherent_fraction:.2f} % of scenarios"
)

print(
    f"Protected mean NIS across scenarios : "
    f"{np.mean(protected_nis_values):.3f}"
)

print(
    f"Permissive mean NIS across scenarios : "
    f"{np.mean(permissive_nis_values):.3f}"
)

print()


# ------------------------------------------------------------
# 21. WORST CASE TABLE
# ------------------------------------------------------------

print(
    "----- WORST CASES -----"
)

for _, row in worst_cases.iterrows():

    print(
        f"{row['worst_case_definition']} : "
        f"{row['scenario_id']} | "
        f"{row['worst_case_metric']} = "
        f"{row[row['worst_case_metric']]:.3f}"
    )

print()


# ------------------------------------------------------------
# 22. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)

print(
    f"Protected better low-red in >75% scenarios : "
    f"{protected_low_better_majority}"
)

print(
    f"Protected better dual outage in >75% scenarios : "
    f"{protected_dual_better_majority}"
)

print(
    f"Protected better mission RMSE in >75% scenarios : "
    f"{protected_mission_better_majority}"
)

print(
    f"Protected NIS coherent in >70% scenarios : "
    f"{protected_nis_majority_coherent}"
)

print()


# ------------------------------------------------------------
# 23. VALIDATION
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
    f"{scenario_order_complete}"
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
    f"Protected blocks all 5-satellite updates : "
    f"{protected_blocks_low_redundancy}"
)

print(
    f"Protected blocks all missed low-red faults : "
    f"{protected_blocks_missed_faults}"
)

print()

print(
    f"VALIDATION GLOBALE 016-B : "
    f"{validation_016b}"
)

print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 24. FIGURE — MISSION RMSE PAIRED
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)


plt.scatter(
    results[
        "permissive_mission_rmse"
    ],

    results[
        "protected_mission_rmse"
    ]
)


maximum_value = max(
    results[
        "permissive_mission_rmse"
    ].max(),

    results[
        "protected_mission_rmse"
    ].max()
)


plt.plot(
    [
        0.0,
        maximum_value
    ],

    [
        0.0,
        maximum_value
    ],

    linestyle="--",
    label="Equal performance"
)


plt.xlabel(
    "Permissive mission RMSE [m]"
)

plt.ylabel(
    "Protected mission RMSE [m]"
)

plt.title(
    "AURORA — 016-B paired mission robustness"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016b_paired_mission_rmse.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 25. FIGURE — LOW REDUNDANCY PAIRED
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)


plt.scatter(
    results[
        "permissive_low_redundancy_rmse"
    ],

    results[
        "protected_low_redundancy_rmse"
    ]
)


maximum_value = max(
    results[
        "permissive_low_redundancy_rmse"
    ].max(),

    results[
        "protected_low_redundancy_rmse"
    ].max()
)


plt.plot(
    [
        0.0,
        maximum_value
    ],

    [
        0.0,
        maximum_value
    ],

    linestyle="--",
    label="Equal performance"
)


plt.xlabel(
    "Permissive low-redundancy RMSE [m]"
)

plt.ylabel(
    "Protected low-redundancy RMSE [m]"
)

plt.title(
    "AURORA — 016-B low-redundancy robustness"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016b_paired_low_redundancy_rmse.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 26. FIGURE — NIS DISTRIBUTION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.hist(
    results[
        "permissive_nis_mean"
    ],
    bins=16,
    alpha=0.6,
    label="Permissive"
)


plt.hist(
    results[
        "protected_nis_mean"
    ],
    bins=16,
    alpha=0.6,
    label="Protected"
)


plt.axvline(
    3.0,
    linestyle="--",
    label="Expected NIS = 3"
)


plt.xlabel(
    "Mean navigation NIS"
)

plt.ylabel(
    "Scenarios"
)

plt.title(
    "AURORA — 016-B navigation consistency robustness"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016b_nis_distribution.png",

    dpi=200
)


plt.show()