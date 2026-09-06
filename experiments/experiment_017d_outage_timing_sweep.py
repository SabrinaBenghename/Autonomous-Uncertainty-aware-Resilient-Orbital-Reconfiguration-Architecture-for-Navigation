from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    spearmanr,
)

import src.validation.outage_phase_sweep as outage_module


# ============================================================
# AURORA
# Experiment 017-D
#
# DUAL-OUTAGE TIMING / ORBITAL-LOCATION SWEEP
#
# Scientific question:
#
# Does AURORA's navigation resilience depend on the point
# in the mission / orbit at which the dual GNSS +
# star-tracker outage begins?
#
#
# 017-C:
#     varied initial orbital phase
#     outage fixed at 60-75 min
#
# 017-D:
#     initial phase fixed
#     outage start time varied
#
#
# FIXED:
#
#     initial true anomaly = 0 deg
#     outage duration       = 15 min
#
#
# OUTAGE WINDOWS:
#
#      15 -  30 min
#      30 -  45 min
#      45 -  60 min
#      60 -  75 min
#      75 -  90 min
#      90 - 105 min
#     105 - 120 min
#     120 - 135 min
#
#
# 8 stochastic replicate blocks
#
# TOTAL:
#
#     8 timings * 8 replicates = 64 full missions
#
#
# IMPORTANT:
#
# This script supports checkpoint/resume.
#
# If the 64 missions have already been completed and the
# results CSV exists, NO mission is repeated.
#
# ============================================================


# ------------------------------------------------------------
# 1. EXPERIMENT DESIGN
# ------------------------------------------------------------

FIXED_INITIAL_TRUE_ANOMALY_DEG = 0.0


OUTAGE_START_TIMES_MIN = [
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
]


DUAL_OUTAGE_DURATION_MIN = 15.0


NUMBER_OF_REPLICATES = 8


BASE_SEED = 170_400


# ------------------------------------------------------------
# 2. OUTPUT DIRECTORIES
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
    "dual_outage_timing_design_017d.csv"
)


results_path = (
    table_directory
    /
    "phase17_017d_dual_outage_timing_sweep.csv"
)


summary_path = (
    table_directory
    /
    "phase17_017d_timing_summary.csv"
)


statistics_path = (
    table_directory
    /
    "phase17_017d_friedman_statistics.csv"
)


correlation_path = (
    table_directory
    /
    "phase17_017d_geometry_correlations.csv"
)


recovery_statistics_path = (
    table_directory
    /
    "phase17_017d_recovery_statistics.csv"
)


# ------------------------------------------------------------
# 3. SCENARIO WRAPPER
# ------------------------------------------------------------

def run_dual_outage_timing_scenario(
    outage_start_min,
    replicate_index,
    scenario_seed,
):

    outage_start_min = float(
        outage_start_min
    )


    outage_end_min = (
        outage_start_min
        +
        DUAL_OUTAGE_DURATION_MIN
    )


    run_function = (
        outage_module
        .run_dual_outage_phase_scenario
    )


    signature = inspect.signature(
        run_function
    )


    parameter_names = set(
        signature.parameters.keys()
    )


    kwargs = {
        "initial_true_anomaly_deg":
            FIXED_INITIAL_TRUE_ANOMALY_DEG,

        "replicate_index":
            int(
                replicate_index
            ),

        "scenario_seed":
            int(
                scenario_seed
            ),
    }


    start_passed_directly = False
    end_passed_directly = False


    if (
        "dual_outage_start_min"
        in parameter_names
    ):

        kwargs[
            "dual_outage_start_min"
        ] = outage_start_min

        start_passed_directly = True


    elif (
        "outage_start_min"
        in parameter_names
    ):

        kwargs[
            "outage_start_min"
        ] = outage_start_min

        start_passed_directly = True


    if (
        "dual_outage_end_min"
        in parameter_names
    ):

        kwargs[
            "dual_outage_end_min"
        ] = outage_end_min

        end_passed_directly = True


    elif (
        "outage_end_min"
        in parameter_names
    ):

        kwargs[
            "outage_end_min"
        ] = outage_end_min

        end_passed_directly = True


    timing_passed_directly = (
        start_passed_directly
        or
        end_passed_directly
    )


    if timing_passed_directly:

        result = run_function(
            **kwargs
        )


    else:

        old_start = getattr(
            outage_module,
            "DUAL_OUTAGE_START_MIN",
            None
        )


        old_end = getattr(
            outage_module,
            "DUAL_OUTAGE_END_MIN",
            None
        )


        outage_module.DUAL_OUTAGE_START_MIN = (
            outage_start_min
        )


        outage_module.DUAL_OUTAGE_END_MIN = (
            outage_end_min
        )


        try:

            result = run_function(
                **kwargs
            )


        finally:

            if old_start is not None:

                outage_module.DUAL_OUTAGE_START_MIN = (
                    old_start
                )


            if old_end is not None:

                outage_module.DUAL_OUTAGE_END_MIN = (
                    old_end
                )


    result = dict(
        result
    )


    result[
        "experiment_outage_start_min"
    ] = outage_start_min


    result[
        "experiment_outage_end_min"
    ] = outage_end_min


    result[
        "experiment_outage_duration_min"
    ] = DUAL_OUTAGE_DURATION_MIN


    result[
        "fixed_initial_true_anomaly_deg"
    ] = (
        FIXED_INITIAL_TRUE_ANOMALY_DEG
    )


    possible_start_columns = [
        "dual_outage_start_min",
        "outage_start_min",
    ]


    possible_end_columns = [
        "dual_outage_end_min",
        "outage_end_min",
    ]


    for column in possible_start_columns:

        if column in result:

            reported_start = float(
                result[
                    column
                ]
            )


            if not np.isclose(
                reported_start,
                outage_start_min,
                atol=1.0e-9,
            ):

                raise RuntimeError(
                    "\n017-D TIMING MISMATCH\n"
                    f"Requested start : "
                    f"{outage_start_min:.3f} min\n"
                    f"Reported start  : "
                    f"{reported_start:.3f} min\n"
                )


    for column in possible_end_columns:

        if column in result:

            reported_end = float(
                result[
                    column
                ]
            )


            if not np.isclose(
                reported_end,
                outage_end_min,
                atol=1.0e-9,
            ):

                raise RuntimeError(
                    "\n017-D TIMING MISMATCH\n"
                    f"Requested end : "
                    f"{outage_end_min:.3f} min\n"
                    f"Reported end  : "
                    f"{reported_end:.3f} min\n"
                )


    return result


# ------------------------------------------------------------
# 4. FIXED CAMPAIGN DESIGN
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


    for outage_start_min in (
        OUTAGE_START_TIMES_MIN
    ):

        outage_end_min = (
            outage_start_min
            +
            DUAL_OUTAGE_DURATION_MIN
        )


        design_rows.append(
            {
                "timing_name":
                    (
                        f"OUTAGE_"
                        f"{int(round(outage_start_min)):03d}_"
                        f"{int(round(outage_end_min)):03d}"
                    ),

                "outage_start_min":
                    float(
                        outage_start_min
                    ),

                "outage_end_min":
                    float(
                        outage_end_min
                    ),

                "outage_duration_min":
                    float(
                        DUAL_OUTAGE_DURATION_MIN
                    ),

                "initial_true_anomaly_deg":
                    float(
                        FIXED_INITIAL_TRUE_ANOMALY_DEG
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
    index=False
)


total_expected_runs = (
    len(
        OUTAGE_START_TIMES_MIN
    )
    *
    NUMBER_OF_REPLICATES
)


design_valid = bool(
    len(
        design
    )
    ==
    total_expected_runs
)


# ------------------------------------------------------------
# 5. CHECKPOINT / RESUME
# ------------------------------------------------------------

if results_path.exists():

    existing_results = pd.read_csv(
        results_path
    )


    if (
        "experiment_outage_start_min"
        in existing_results.columns
        and
        "replicate_index"
        in existing_results.columns
    ):

        existing_results = (
            existing_results
            .sort_values(
                [
                    "replicate_index",
                    "experiment_outage_start_min",
                ]
            )
            .drop_duplicates(
                subset=[
                    "experiment_outage_start_min",
                    "replicate_index",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )


        existing_results.to_csv(
            results_path,
            index=False
        )


        completed_keys = set(
            zip(
                existing_results[
                    "experiment_outage_start_min"
                ].astype(
                    float
                ),

                existing_results[
                    "replicate_index"
                ].astype(
                    int
                ),
            )
        )


    else:

        raise RuntimeError(
            "\nExisting 017-D results file does not contain "
            "the expected checkpoint columns.\n"
            f"File: {results_path}\n"
        )


else:

    existing_results = pd.DataFrame()

    completed_keys = set()


remaining_mask = []


for _, row in design.iterrows():

    key = (
        float(
            row[
                "outage_start_min"
            ]
        ),

        int(
            row[
                "replicate_index"
            ]
        ),
    )


    remaining_mask.append(
        key
        not in
        completed_keys
    )


remaining_design = (
    design[
        remaining_mask
    ]
    .copy()
)


# ------------------------------------------------------------
# 6. HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experiment 017-D"
)


print(
    "Dual-outage timing / orbital-location sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Fixed initial true anomaly : "
    f"{FIXED_INITIAL_TRUE_ANOMALY_DEG:.1f} deg"
)


print(
    f"Outage start times : "
    f"{OUTAGE_START_TIMES_MIN} min"
)


print(
    f"Fixed outage duration : "
    f"{DUAL_OUTAGE_DURATION_MIN:.1f} min"
)


print(
    f"Replicates per timing : "
    f"{NUMBER_OF_REPLICATES}"
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


if len(
    remaining_design
) == 0:

    print(
        "\nCheckpoint complete — "
        "NO mission simulations will be repeated."
    )


print()


# ------------------------------------------------------------
# 7. RUN ONLY MISSING MISSIONS
# ------------------------------------------------------------

new_rows = []


for local_index, (_, row) in enumerate(
    remaining_design.iterrows()
):

    outage_start_min = float(
        row[
            "outage_start_min"
        ]
    )


    outage_end_min = float(
        row[
            "outage_end_min"
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
        f"[{global_index:02d}/"
        f"{total_expected_runs}] "
        f"outage="
        f"{outage_start_min:6.1f}-"
        f"{outage_end_min:6.1f} min | "
        f"rep={replicate_index:02d} | "
        f"seed={scenario_seed}"
    )


    result = (
        run_dual_outage_timing_scenario(
            outage_start_min=
                outage_start_min,

            replicate_index=
                replicate_index,

            scenario_seed=
                scenario_seed,
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
                ),
            ),
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
                "experiment_outage_start_min",
            ]
        )
        .drop_duplicates(
            subset=[
                "experiment_outage_start_min",
                "replicate_index",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


    checkpoint.to_csv(
        results_path,
        index=False
    )


    post_value = result.get(
        "post_recovery_rmse_m",
        np.nan
    )


    if np.isfinite(
        post_value
    ):

        post_text = (
            f"{post_value:.3f} m"
        )

    else:

        post_text = "N/A"


    print(
        f"    PDOPpre="
        f"{result['pre_outage_pdop']:.3f} | "
        f"RMSE outage="
        f"{result['outage_rmse_m']:.3f} m | "
        f"end="
        f"{result['outage_end_error_m']:.3f} m | "
        f"sigma_end="
        f"{result['outage_end_sigma_m']:.3f} m | "
        f"post="
        f"{post_text} | "
        f"att="
        f"{result['attitude_outage_rmse_deg']:.4f} deg"
    )


# ------------------------------------------------------------
# 8. RELOAD FINAL CHECKPOINT
# ------------------------------------------------------------

if not results_path.exists():

    raise RuntimeError(
        "017-D results file does not exist."
    )


results = pd.read_csv(
    results_path
)


results = (
    results
    .sort_values(
        [
            "replicate_index",
            "experiment_outage_start_min",
        ]
    )
    .drop_duplicates(
        subset=[
            "experiment_outage_start_min",
            "replicate_index",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 9. CORE VALIDATION
# ------------------------------------------------------------

all_runs_complete = bool(
    len(
        results
    )
    ==
    total_expected_runs
)


unique_keys = bool(
    results[
        [
            "experiment_outage_start_min",
            "replicate_index",
        ]
    ]
    .drop_duplicates()
    .shape[
        0
    ]
    ==
    total_expected_runs
)


all_timings_complete = bool(
    np.all(
        results.groupby(
            "experiment_outage_start_min"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)


all_replicates_cover_all_timings = bool(
    np.all(
        results.groupby(
            "replicate_index"
        )[
            "experiment_outage_start_min"
        ]
        .nunique()
        ==
        len(
            OUTAGE_START_TIMES_MIN
        )
    )
)


fixed_initial_phase_valid = bool(
    np.allclose(
        results[
            "fixed_initial_true_anomaly_deg"
        ].to_numpy(
            dtype=float
        ),

        FIXED_INITIAL_TRUE_ANOMALY_DEG,
    )
)


experiment_outage_duration_valid = bool(
    np.allclose(
        (
            results[
                "experiment_outage_end_min"
            ].to_numpy(
                dtype=float
            )
            -
            results[
                "experiment_outage_start_min"
            ].to_numpy(
                dtype=float
            )
        ),

        DUAL_OUTAGE_DURATION_MIN,
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
# IMPORTANT:
#
# pre_outage_rmse_m and post_recovery_rmse_m are diagnostic
# metrics only.
#
# They are NOT required to be finite for all timings because
# the early/late mission windows do not always provide the
# complete evaluation interval required by those metrics.
# ------------------------------------------------------------

core_columns = [
    "pre_outage_pdop",
    "mission_rmse_m",
    "outage_rmse_m",
    "outage_start_error_m",
    "outage_end_error_m",
    "outage_start_sigma_m",
    "outage_end_sigma_m",
    "final_error_m",
    "navigation_nis_mean",
    "navigation_nis_coverage_percent",
    "attitude_outage_rmse_deg",
    "attitude_end_outage_deg",
    "healthy_false_alarm_rate",
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


# ------------------------------------------------------------
# 10. DIAGNOSTIC METRIC AVAILABILITY
# ------------------------------------------------------------

pre_outage_rmse_available_runs = int(
    results[
        "pre_outage_rmse_m"
    ]
    .notna()
    .sum()
)


pre_outage_rmse_coverage_percent = float(
    100.0
    *
    pre_outage_rmse_available_runs
    /
    max(
        len(
            results
        ),
        1
    )
)


post_recovery_available_runs = int(
    results[
        "post_recovery_rmse_m"
    ]
    .notna()
    .sum()
)


post_recovery_total_runs = int(
    len(
        results
    )
)


post_recovery_coverage_percent = float(
    100.0
    *
    post_recovery_available_runs
    /
    max(
        post_recovery_total_runs,
        1
    )
)


# ------------------------------------------------------------
# 11. TIMING SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        "experiment_outage_start_min",
        as_index=False,
    )
    .agg(
        runs=(
            "replicate_index",
            "size",
        ),

        visible_mean=(
            "natural_visible_satellites_mean",
            "mean",
        ),

        pre_outage_pdop_mean=(
            "pre_outage_pdop",
            "mean",
        ),

        pre_outage_pdop_std=(
            "pre_outage_pdop",
            "std",
        ),

        pre_outage_update_age_mean_s=(
            "pre_outage_update_age_s",
            "mean",
        ),

        pre_outage_rmse_mean=(
            "pre_outage_rmse_m",
            "mean",
        ),

        pre_outage_rmse_std=(
            "pre_outage_rmse_m",
            "std",
        ),

        pre_outage_rmse_valid_runs=(
            "pre_outage_rmse_m",
            "count",
        ),

        outage_start_error_mean=(
            "outage_start_error_m",
            "mean",
        ),

        outage_rmse_mean=(
            "outage_rmse_m",
            "mean",
        ),

        outage_rmse_std=(
            "outage_rmse_m",
            "std",
        ),

        outage_end_error_mean=(
            "outage_end_error_m",
            "mean",
        ),

        outage_end_error_std=(
            "outage_end_error_m",
            "std",
        ),

        outage_start_sigma_mean=(
            "outage_start_sigma_m",
            "mean",
        ),

        outage_end_sigma_mean=(
            "outage_end_sigma_m",
            "mean",
        ),

        post_recovery_rmse_mean=(
            "post_recovery_rmse_m",
            "mean",
        ),

        post_recovery_rmse_std=(
            "post_recovery_rmse_m",
            "std",
        ),

        post_recovery_valid_runs=(
            "post_recovery_rmse_m",
            "count",
        ),

        attitude_outage_rmse_mean_deg=(
            "attitude_outage_rmse_deg",
            "mean",
        ),

        navigation_nis_mean=(
            "navigation_nis_mean",
            "mean",
        ),

        healthy_false_alarm_mean=(
            "healthy_false_alarm_rate",
            "mean",
        ),
    )
)


summary[
    "experiment_outage_end_min"
] = (
    summary[
        "experiment_outage_start_min"
    ]
    +
    DUAL_OUTAGE_DURATION_MIN
)


summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 12. MAIN FRIEDMAN TEST
# ------------------------------------------------------------

FRIEDMAN_METRICS = [
    "outage_rmse_m",
    "outage_end_error_m",
    "pre_outage_pdop",
]


friedman_rows = []


for metric in FRIEDMAN_METRICS:

    pivot = (
        results.pivot(
            index=
                "replicate_index",

            columns=
                "experiment_outage_start_min",

            values=
                metric,
        )
        .reindex(
            columns=
                OUTAGE_START_TIMES_MIN
        )
    )


    if pivot.isna().any().any():

        raise RuntimeError(
            "Incomplete repeated-measures "
            f"matrix for primary metric: {metric}"
        )


    timing_samples = [
        pivot[
            outage_start_min
        ].to_numpy(
            dtype=float
        )

        for outage_start_min
        in OUTAGE_START_TIMES_MIN
    ]


    statistic, p_value = (
        friedmanchisquare(
            *timing_samples
        )
    )


    number_of_blocks = (
        NUMBER_OF_REPLICATES
    )


    number_of_timings = len(
        OUTAGE_START_TIMES_MIN
    )


    kendalls_w = float(
        statistic
        /
        (
            number_of_blocks
            *
            (
                number_of_timings
                -
                1
            )
        )
    )


    timing_means = np.array(
        [
            np.mean(
                sample
            )

            for sample in timing_samples
        ],
        dtype=float,
    )


    mean_range = float(
        np.max(
            timing_means
        )
        -
        np.min(
            timing_means
        )
    )


    overall_mean = float(
        np.mean(
            timing_means
        )
    )


    relative_range_percent = float(
        100.0
        *
        mean_range
        /
        max(
            abs(
                overall_mean
            ),
            1.0e-12,
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
                kendalls_w,

            "timing_mean_range":
                mean_range,

            "relative_timing_range_percent":
                relative_range_percent,

            "significant_0p05":
                bool(
                    p_value
                    <
                    0.05
                ),
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
# 13. POST-RECOVERY ANALYSIS
# ------------------------------------------------------------

recovery_counts = (
    results.groupby(
        "experiment_outage_start_min"
    )[
        "post_recovery_rmse_m"
    ]
    .apply(
        lambda series:
            int(
                np.isfinite(
                    series.to_numpy(
                        dtype=float
                    )
                ).sum()
            )
    )
)


complete_recovery_timings = [
    float(
        timing
    )

    for timing, count in recovery_counts.items()

    if int(
        count
    )
    ==
    NUMBER_OF_REPLICATES
]


recovery_friedman_valid = False
recovery_friedman_statistic = np.nan
recovery_friedman_p_value = np.nan
recovery_kendalls_w = np.nan


if len(
    complete_recovery_timings
) >= 3:

    recovery_pivot = (
        results[
            results[
                "experiment_outage_start_min"
            ].isin(
                complete_recovery_timings
            )
        ]
        .pivot(
            index=
                "replicate_index",

            columns=
                "experiment_outage_start_min",

            values=
                "post_recovery_rmse_m",
        )
        .reindex(
            columns=
                complete_recovery_timings
        )
    )


    if not recovery_pivot.isna().any().any():

        recovery_samples = [
            recovery_pivot[
                timing
            ].to_numpy(
                dtype=float
            )

            for timing in complete_recovery_timings
        ]


        (
            recovery_friedman_statistic,
            recovery_friedman_p_value,
        ) = friedmanchisquare(
            *recovery_samples
        )


        recovery_kendalls_w = float(
            recovery_friedman_statistic
            /
            (
                NUMBER_OF_REPLICATES
                *
                (
                    len(
                        complete_recovery_timings
                    )
                    -
                    1
                )
            )
        )


        recovery_friedman_valid = True


recovery_table = pd.DataFrame(
    [
        {
            "complete_recovery_timings":
                ",".join(
                    [
                        f"{timing:.1f}"
                        for timing in complete_recovery_timings
                    ]
                ),

            "number_of_complete_timings":
                len(
                    complete_recovery_timings
                ),

            "friedman_valid":
                recovery_friedman_valid,

            "friedman_statistic":
                recovery_friedman_statistic,

            "p_value":
                recovery_friedman_p_value,

            "kendalls_w":
                recovery_kendalls_w,

            "significant_0p05":
                bool(
                    recovery_friedman_valid
                    and
                    recovery_friedman_p_value
                    <
                    0.05
                ),
        }
    ]
)


recovery_table.to_csv(
    recovery_statistics_path,
    index=False
)


# ------------------------------------------------------------
# 14. CORRELATION ANALYSIS
# ------------------------------------------------------------

CORRELATION_PAIRS = [
    (
        "pre_outage_pdop",
        "outage_rmse_m",
    ),

    (
        "pre_outage_pdop",
        "outage_end_error_m",
    ),

    (
        "pre_outage_pdop",
        "post_recovery_rmse_m",
    ),

    (
        "natural_visible_satellites_mean",
        "outage_rmse_m",
    ),

    (
        "pre_outage_rmse_m",
        "outage_rmse_m",
    ),

    (
        "outage_start_sigma_m",
        "outage_rmse_m",
    ),

    (
        "outage_start_sigma_m",
        "outage_end_error_m",
    ),
]


correlation_rows = []


for predictor, outcome in CORRELATION_PAIRS:

    predictor_values = (
        results[
            predictor
        ]
        .to_numpy(
            dtype=float
        )
    )


    outcome_values = (
        results[
            outcome
        ]
        .to_numpy(
            dtype=float
        )
    )


    finite_mask = (
        np.isfinite(
            predictor_values
        )
        &
        np.isfinite(
            outcome_values
        )
    )


    number_of_valid_pairs = int(
        np.sum(
            finite_mask
        )
    )


    if number_of_valid_pairs >= 3:

        rho, p_value = spearmanr(
            predictor_values[
                finite_mask
            ],

            outcome_values[
                finite_mask
            ],
        )


    else:

        rho = np.nan
        p_value = np.nan


    correlation_rows.append(
        {
            "predictor":
                predictor,

            "outcome":
                outcome,

            "n_valid":
                number_of_valid_pairs,

            "spearman_rho":
                float(
                    rho
                ),

            "p_value":
                float(
                    p_value
                ),

            "significant_0p05":
                bool(
                    np.isfinite(
                        p_value
                    )
                    and
                    p_value
                    <
                    0.05
                ),
        }
    )


correlation_table = pd.DataFrame(
    correlation_rows
)


correlation_table.to_csv(
    correlation_path,
    index=False
)


# ------------------------------------------------------------
# 15. BEST / WORST TIMINGS
# ------------------------------------------------------------

best_index = (
    summary[
        "outage_rmse_mean"
    ]
    .idxmin()
)


worst_index = (
    summary[
        "outage_rmse_mean"
    ]
    .idxmax()
)


best_timing = (
    summary.loc[
        best_index
    ]
)


worst_timing = (
    summary.loc[
        worst_index
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
# 16. SCIENTIFIC FLAGS
# ------------------------------------------------------------

outage_timing_changes_rmse = bool(
    outage_metric_row[
        "p_value"
    ]
    <
    0.05
)


outage_timing_changes_end_error = bool(
    end_error_metric_row[
        "p_value"
    ]
    <
    0.05
)


outage_timing_effect_large = bool(
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
# 17. GLOBAL VALIDATION
# ------------------------------------------------------------

validation_017d = bool(
    design_valid
    and
    all_runs_complete
    and
    unique_keys
    and
    all_timings_complete
    and
    all_replicates_cover_all_timings
    and
    fixed_initial_phase_valid
    and
    experiment_outage_duration_valid
    and
    time_sync_valid
    and
    all_core_metrics_finite
)


# ------------------------------------------------------------
# 18. PRINT RESULTS
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)


print(
    "AURORA — EXPERIMENT 017-D RESULTS"
)


print(
    "Dual-outage timing / orbital-location sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)


print(
    f"Outage timings : "
    f"{len(OUTAGE_START_TIMES_MIN)}"
)


print(
    f"Replicates per timing : "
    f"{NUMBER_OF_REPLICATES}"
)


print(
    f"Fixed initial true anomaly : "
    f"{FIXED_INITIAL_TRUE_ANOMALY_DEG:.1f} deg"
)


print(
    f"Fixed dual-outage duration : "
    f"{DUAL_OUTAGE_DURATION_MIN:.1f} min"
)


print()


# ------------------------------------------------------------
# 19. TIMING SUMMARY
# ------------------------------------------------------------

print(
    "----- TIMING SUMMARY -----"
)


formatters = {}


for column in summary.columns:

    if column not in [
        "runs",
        "pre_outage_rmse_valid_runs",
        "post_recovery_valid_runs",
    ]:

        formatters[
            column
        ] = (
            lambda value:
                (
                    f"{value:.4f}"
                    if pd.notna(
                        value
                    )
                    else
                    "NaN"
                )
        )


print(
    summary.to_string(
        index=False,
        formatters=formatters,
    )
)


print()


# ------------------------------------------------------------
# 20. FRIEDMAN RESULTS
# ------------------------------------------------------------

print(
    "----- REPEATED-MEASURES TIMING TEST -----"
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

            "timing_mean_range":
                lambda value:
                    f"{value:.4f}",

            "relative_timing_range_percent":
                lambda value:
                    f"{value:.2f}",
        },
    )
)


print()


# ------------------------------------------------------------
# 21. DIAGNOSTIC AVAILABILITY
# ------------------------------------------------------------

print(
    "----- DIAGNOSTIC METRIC AVAILABILITY -----"
)


print(
    f"Finite pre-outage RMSE runs : "
    f"{pre_outage_rmse_available_runs}/"
    f"{len(results)} "
    f"("
    f"{pre_outage_rmse_coverage_percent:.1f}%"
    f")"
)


print(
    f"Finite post-recovery runs : "
    f"{post_recovery_available_runs}/"
    f"{post_recovery_total_runs} "
    f"("
    f"{post_recovery_coverage_percent:.1f}%"
    f")"
)


print()


# ------------------------------------------------------------
# 22. POST-RECOVERY ANALYSIS
# ------------------------------------------------------------

print(
    "----- POST-RECOVERY ANALYSIS -----"
)


print(
    f"Timings with complete recovery data : "
    f"{complete_recovery_timings}"
)


if recovery_friedman_valid:

    print(
        f"Recovery Friedman statistic : "
        f"{recovery_friedman_statistic:.3f}"
    )


    print(
        f"Recovery Friedman p-value : "
        f"{recovery_friedman_p_value:.6f}"
    )


    print(
        f"Recovery Kendall W : "
        f"{recovery_kendalls_w:.3f}"
    )


else:

    print(
        "Recovery Friedman test : "
        "not available"
    )


print()


# ------------------------------------------------------------
# 23. CORRELATIONS
# ------------------------------------------------------------

print(
    "----- GEOMETRY / STATE / ROBUSTNESS CORRELATIONS -----"
)


print(
    correlation_table.to_string(
        index=False,
        formatters={
            "spearman_rho":
                lambda value:
                    (
                        f"{value:.4f}"
                        if np.isfinite(
                            value
                        )
                        else
                        "NaN"
                    ),

            "p_value":
                lambda value:
                    (
                        f"{value:.6f}"
                        if np.isfinite(
                            value
                        )
                        else
                        "NaN"
                    ),
        },
    )
)


print()


# ------------------------------------------------------------
# 24. BEST / WORST
# ------------------------------------------------------------

print(
    "----- BEST / WORST OUTAGE TIMING -----"
)


print(
    f"Best timing by outage RMSE : "
    f"{best_timing['experiment_outage_start_min']:.1f}-"
    f"{best_timing['experiment_outage_end_min']:.1f} min | "
    f"RMSE="
    f"{best_timing['outage_rmse_mean']:.3f} m"
)


print(
    f"Worst timing by outage RMSE : "
    f"{worst_timing['experiment_outage_start_min']:.1f}-"
    f"{worst_timing['experiment_outage_end_min']:.1f} min | "
    f"RMSE="
    f"{worst_timing['outage_rmse_mean']:.3f} m"
)


print(
    f"Mean timing range : "
    f"{outage_metric_row['timing_mean_range']:.3f} m "
    f"("
    f"{outage_metric_row['relative_timing_range_percent']:.1f}%"
    f")"
)


print()


# ------------------------------------------------------------
# 25. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Outage timing significantly changes outage RMSE : "
    f"{outage_timing_changes_rmse}"
)


print(
    f"Outage timing significantly changes "
    f"end-of-outage error : "
    f"{outage_timing_changes_end_error}"
)


print(
    f"Outage timing effect Kendall W >= 0.30 : "
    f"{outage_timing_effect_large}"
)


print(
    f"Protected NIS remains in [1.5, 4.5] "
    f"for every timing : "
    f"{all_nis_reasonable}"
)


print()


# ------------------------------------------------------------
# 26. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"Fixed design valid : "
    f"{design_valid}"
)


print(
    f"All {total_expected_runs} missions complete : "
    f"{all_runs_complete}"
)


print(
    f"All timing/replicate keys unique : "
    f"{unique_keys}"
)


print(
    f"{NUMBER_OF_REPLICATES} replicates "
    f"available for every timing : "
    f"{all_timings_complete}"
)


print(
    f"Every replicate covers all "
    f"{len(OUTAGE_START_TIMES_MIN)} timings : "
    f"{all_replicates_cover_all_timings}"
)


print(
    f"Initial orbital phase fixed : "
    f"{fixed_initial_phase_valid}"
)


print(
    f"Dual-outage duration fixed at "
    f"{DUAL_OUTAGE_DURATION_MIN:.1f} min : "
    f"{experiment_outage_duration_valid}"
)


print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)


print(
    f"All required core metrics finite : "
    f"{all_core_metrics_finite}"
)


print(
    f"Pre-outage RMSE coverage : "
    f"{pre_outage_rmse_coverage_percent:.1f}% "
    f"(diagnostic only)"
)


print(
    f"Post-recovery RMSE coverage : "
    f"{post_recovery_coverage_percent:.1f}% "
    f"(diagnostic only)"
)


print()


print(
    f"VALIDATION GLOBALE 017-D : "
    f"{validation_017d}"
)


print(
    "========================================================================================================================"
)


# ============================================================
# 27. FIGURE — OUTAGE RMSE VS TIMING
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_outage_start_min"
    ],

    summary[
        "outage_rmse_mean"
    ],

    yerr=
        summary[
            "outage_rmse_std"
        ],

    marker="o",
)


plt.xlabel(
    "Dual-outage start time [min]"
)


plt.ylabel(
    "Protected dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-D outage resilience vs outage timing"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_outage_rmse_vs_timing.png",

    dpi=200,
)


plt.show()


# ============================================================
# 28. FIGURE — END-OF-OUTAGE ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_outage_start_min"
    ],

    summary[
        "outage_end_error_mean"
    ],

    yerr=
        summary[
            "outage_end_error_std"
        ],

    marker="o",
)


plt.xlabel(
    "Dual-outage start time [min]"
)


plt.ylabel(
    "Position error at end of dual outage [m]"
)


plt.title(
    "AURORA — 017-D end-of-outage error vs outage timing"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_end_error_vs_timing.png",

    dpi=200,
)


plt.show()


# ============================================================
# 29. FIGURE — PRE-OUTAGE PDOP
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_outage_start_min"
    ],

    summary[
        "pre_outage_pdop_mean"
    ],

    yerr=
        summary[
            "pre_outage_pdop_std"
        ],

    marker="o",
)


plt.xlabel(
    "Dual-outage start time [min]"
)


plt.ylabel(
    "PDOP immediately before outage"
)


plt.title(
    "AURORA — 017-D GNSS geometry at outage onset"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_pre_outage_pdop_vs_timing.png",

    dpi=200,
)


plt.show()


# ============================================================
# 30. FIGURE — PDOP VS OUTAGE RMSE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    results[
        "pre_outage_pdop"
    ],

    results[
        "outage_rmse_m"
    ],
)


plt.xlabel(
    "PDOP immediately before outage"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-D GNSS geometry vs outage degradation"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_pdop_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 31. FIGURE — START SIGMA VS OUTAGE RMSE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    results[
        "outage_start_sigma_m"
    ],

    results[
        "outage_rmse_m"
    ],
)


plt.xlabel(
    "Position sigma at outage onset [m]"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-D pre-outage uncertainty vs outage degradation"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_start_sigma_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 32. FIGURE — END SIGMA VS TIMING
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    summary[
        "experiment_outage_start_min"
    ],

    summary[
        "outage_end_sigma_mean"
    ],

    marker="o",
)


plt.xlabel(
    "Dual-outage start time [min]"
)


plt.ylabel(
    "Mean position sigma at outage end [m]"
)


plt.title(
    "AURORA — 017-D uncertainty at end of outage"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_end_sigma_vs_timing.png",

    dpi=200,
)


plt.show()


# ============================================================
# 33. FIGURE — SIGMA GROWTH DURING OUTAGE
# ============================================================

sigma_growth_mean = (
    summary[
        "outage_end_sigma_mean"
    ]
    -
    summary[
        "outage_start_sigma_mean"
    ]
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    summary[
        "experiment_outage_start_min"
    ],

    sigma_growth_mean,

    marker="o",
)


plt.xlabel(
    "Dual-outage start time [min]"
)


plt.ylabel(
    "Mean sigma growth during outage [m]"
)


plt.title(
    "AURORA — 017-D uncertainty growth vs outage timing"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017d_sigma_growth_vs_timing.png",

    dpi=200,
)


plt.show()


# ============================================================
# 34. FIGURE — POST-RECOVERY RMSE
# ============================================================

recovery_plot_data = (
    summary[
        summary[
            "post_recovery_valid_runs"
        ]
        >
        0
    ]
    .copy()
)


if len(
    recovery_plot_data
) > 0:

    plt.figure(
        figsize=(10, 6)
    )


    plt.errorbar(
        recovery_plot_data[
            "experiment_outage_start_min"
        ],

        recovery_plot_data[
            "post_recovery_rmse_mean"
        ],

        yerr=
            recovery_plot_data[
                "post_recovery_rmse_std"
            ],

        marker="o",
    )


    plt.xlabel(
        "Dual-outage start time [min]"
    )


    plt.ylabel(
        "Post-recovery RMSE [m]"
    )


    plt.title(
        "AURORA — 017-D post-recovery performance"
    )


    plt.grid(
        True
    )


    plt.tight_layout()


    plt.savefig(
        figure_directory
        /
        "phase17_017d_post_recovery_rmse.png",

        dpi=200,
    )


    plt.show()