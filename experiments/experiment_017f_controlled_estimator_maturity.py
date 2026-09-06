from pathlib import Path
import inspect

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
# AURORA
# Experiment 017-F
#
# CONTROLLED ESTIMATOR-MATURITY INTERVENTION
#
# Scientific question:
#
# Does estimator maturity causally influence resilience to a
# fixed dual GNSS + star-tracker outage?
#
# FIXED ACROSS ALL CONDITIONS:
#
#   initial true anomaly = 0 deg
#   dual outage          = 60-75 min
#   outage duration      = 15 min
#   stochastic seed      = paired across estimator ages
#
# MANIPULATED FACTOR:
#
#   estimator age at outage onset
#
#   5, 15, 30, 45, 60 min
#
# Equivalent estimator start times:
#
#   55, 45, 30, 15, 0 min
#
# 5 maturity levels * 8 stochastic replicates = 40 missions
#
# IMPORTANT:
#
# This script REFUSES to run if it cannot find a real
# estimator-start / estimator-age control in the simulation
# interface. It will never silently pretend maturity changed.
# ============================================================


# ------------------------------------------------------------
# 1. EXPERIMENT DESIGN
# ------------------------------------------------------------

FIXED_INITIAL_TRUE_ANOMALY_DEG = 0.0

FIXED_OUTAGE_START_MIN = 60.0
FIXED_OUTAGE_END_MIN = 75.0
FIXED_OUTAGE_DURATION_MIN = 15.0

ESTIMATOR_AGES_MIN = [
    5.0,
    15.0,
    30.0,
    45.0,
    60.0,
]

NUMBER_OF_REPLICATES = 8
BASE_SEED = 170_500


# ------------------------------------------------------------
# 2. OUTPUT PATHS
# ------------------------------------------------------------

table_directory = Path("results") / "tables"
figure_directory = Path("results") / "figures"
data_directory = Path("data") / "phase17"

table_directory.mkdir(parents=True, exist_ok=True)
figure_directory.mkdir(parents=True, exist_ok=True)
data_directory.mkdir(parents=True, exist_ok=True)

design_path = (
    data_directory
    / "controlled_estimator_maturity_design_017f.csv"
)

results_path = (
    table_directory
    / "phase17_017f_controlled_estimator_maturity.csv"
)

summary_path = (
    table_directory
    / "phase17_017f_maturity_summary.csv"
)

statistics_path = (
    table_directory
    / "phase17_017f_friedman_statistics.csv"
)

correlation_path = (
    table_directory
    / "phase17_017f_correlations.csv"
)

posthoc_path = (
    table_directory
    / "phase17_017f_posthoc_wilcoxon.csv"
)

validation_path = (
    table_directory
    / "phase17_017f_validation.csv"
)


# ------------------------------------------------------------
# 3. ESTIMATOR-CONTROL DISCOVERY
# ------------------------------------------------------------

run_function = outage_module.run_dual_outage_phase_scenario

run_signature = inspect.signature(
    run_function
)

run_parameters = set(
    run_signature.parameters.keys()
)


CONTROL_CANDIDATES = [
    ("estimator_start_min", "start_min"),
    ("estimator_start_time_min", "start_min"),
    ("navigation_estimator_start_min", "start_min"),
    ("navigation_start_min", "start_min"),
    ("filter_start_min", "start_min"),
    ("filter_start_time_min", "start_min"),
    ("ekf_start_min", "start_min"),
    ("ekf_start_time_min", "start_min"),
    ("estimator_activation_min", "start_min"),
    ("navigation_activation_min", "start_min"),

    ("estimator_age_min", "age_min"),
    ("filter_age_min", "age_min"),
    ("ekf_age_min", "age_min"),
    ("estimator_warmup_min", "age_min"),
    ("filter_warmup_min", "age_min"),
    ("navigation_warmup_min", "age_min"),

    ("estimator_start_s", "start_s"),
    ("estimator_start_time_s", "start_s"),
    ("navigation_estimator_start_s", "start_s"),
    ("navigation_start_s", "start_s"),
    ("filter_start_s", "start_s"),
    ("filter_start_time_s", "start_s"),
    ("ekf_start_s", "start_s"),
    ("estimator_activation_s", "start_s"),

    ("estimator_age_s", "age_s"),
    ("filter_age_s", "age_s"),
    ("ekf_age_s", "age_s"),
    ("estimator_warmup_s", "age_s"),
    ("filter_warmup_s", "age_s"),
]


MODULE_GLOBAL_CANDIDATES = [
    ("ESTIMATOR_START_MIN", "start_min"),
    ("ESTIMATOR_START_TIME_MIN", "start_min"),
    ("NAVIGATION_ESTIMATOR_START_MIN", "start_min"),
    ("NAVIGATION_START_MIN", "start_min"),
    ("FILTER_START_MIN", "start_min"),
    ("FILTER_START_TIME_MIN", "start_min"),
    ("EKF_START_MIN", "start_min"),
    ("EKF_START_TIME_MIN", "start_min"),
    ("ESTIMATOR_ACTIVATION_MIN", "start_min"),
    ("NAVIGATION_ACTIVATION_MIN", "start_min"),

    ("ESTIMATOR_AGE_MIN", "age_min"),
    ("FILTER_AGE_MIN", "age_min"),
    ("EKF_AGE_MIN", "age_min"),
    ("ESTIMATOR_WARMUP_MIN", "age_min"),
    ("FILTER_WARMUP_MIN", "age_min"),
    ("NAVIGATION_WARMUP_MIN", "age_min"),

    ("ESTIMATOR_START_S", "start_s"),
    ("ESTIMATOR_START_TIME_S", "start_s"),
    ("NAVIGATION_ESTIMATOR_START_S", "start_s"),
    ("NAVIGATION_START_S", "start_s"),
    ("FILTER_START_S", "start_s"),
    ("FILTER_START_TIME_S", "start_s"),
    ("EKF_START_S", "start_s"),
    ("ESTIMATOR_ACTIVATION_S", "start_s"),

    ("ESTIMATOR_AGE_S", "age_s"),
    ("FILTER_AGE_S", "age_s"),
    ("EKF_AGE_S", "age_s"),
    ("ESTIMATOR_WARMUP_S", "age_s"),
    ("FILTER_WARMUP_S", "age_s"),
]


def discover_estimator_control():

    for name, mode in CONTROL_CANDIDATES:

        if name in run_parameters:

            return {
                "kind":
                    "function_parameter",

                "name":
                    name,

                "mode":
                    mode,
            }


    for name, mode in MODULE_GLOBAL_CANDIDATES:

        if hasattr(
            outage_module,
            name
        ):

            return {
                "kind":
                    "module_global",

                "name":
                    name,

                "mode":
                    mode,
            }


    return None


estimator_control = (
    discover_estimator_control()
)


# ------------------------------------------------------------
# 4. CONTROL VALUE CONVERSION
# ------------------------------------------------------------

def control_value_for_age(
    estimator_age_min,
    mode,
):

    estimator_age_min = float(
        estimator_age_min
    )


    estimator_start_min = (
        FIXED_OUTAGE_START_MIN
        -
        estimator_age_min
    )


    if estimator_start_min < -1.0e-12:

        raise ValueError(
            "Estimator age cannot exceed "
            "outage start time."
        )


    if mode == "start_min":

        return float(
            estimator_start_min
        )


    if mode == "age_min":

        return float(
            estimator_age_min
        )


    if mode == "start_s":

        return float(
            estimator_start_min
            *
            60.0
        )


    if mode == "age_s":

        return float(
            estimator_age_min
            *
            60.0
        )


    raise ValueError(
        f"Unsupported estimator control mode: "
        f"{mode}"
    )


# ------------------------------------------------------------
# 5. FIXED-OUTAGE HELPERS
# ------------------------------------------------------------

OUTAGE_PARAMETER_START_NAMES = [
    "dual_outage_start_min",
    "outage_start_min",
]


OUTAGE_PARAMETER_END_NAMES = [
    "dual_outage_end_min",
    "outage_end_min",
]


def add_fixed_outage_kwargs(
    kwargs
):

    for name in (
        OUTAGE_PARAMETER_START_NAMES
    ):

        if name in run_parameters:

            kwargs[
                name
            ] = (
                FIXED_OUTAGE_START_MIN
            )

            break


    for name in (
        OUTAGE_PARAMETER_END_NAMES
    ):

        if name in run_parameters:

            kwargs[
                name
            ] = (
                FIXED_OUTAGE_END_MIN
            )

            break


    return kwargs


# ------------------------------------------------------------
# 6. CONTROLLED SCENARIO WRAPPER
# ------------------------------------------------------------

def run_controlled_maturity_scenario(
    estimator_age_min,
    replicate_index,
    scenario_seed,
):

    if estimator_control is None:

        raise RuntimeError(
            "No estimator-start / estimator-age "
            "control was found."
        )


    estimator_age_min = float(
        estimator_age_min
    )


    estimator_start_min = (
        FIXED_OUTAGE_START_MIN
        -
        estimator_age_min
    )


    control_value = (
        control_value_for_age(
            estimator_age_min,
            estimator_control[
                "mode"
            ],
        )
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


    kwargs = add_fixed_outage_kwargs(
        kwargs
    )


    old_outage_start = getattr(
        outage_module,
        "DUAL_OUTAGE_START_MIN",
        None,
    )


    old_outage_end = getattr(
        outage_module,
        "DUAL_OUTAGE_END_MIN",
        None,
    )


    old_estimator_control = None


    if (
        estimator_control[
            "kind"
        ]
        ==
        "function_parameter"
    ):

        kwargs[
            estimator_control[
                "name"
            ]
        ] = control_value


    elif (
        estimator_control[
            "kind"
        ]
        ==
        "module_global"
    ):

        old_estimator_control = getattr(
            outage_module,
            estimator_control[
                "name"
            ],
        )


        setattr(
            outage_module,
            estimator_control[
                "name"
            ],
            control_value,
        )


    else:

        raise RuntimeError(
            "Unknown estimator-control "
            "interface kind."
        )


    if (
        "dual_outage_start_min"
        not in kwargs
        and
        "outage_start_min"
        not in kwargs
    ):

        outage_module.DUAL_OUTAGE_START_MIN = (
            FIXED_OUTAGE_START_MIN
        )


    if (
        "dual_outage_end_min"
        not in kwargs
        and
        "outage_end_min"
        not in kwargs
    ):

        outage_module.DUAL_OUTAGE_END_MIN = (
            FIXED_OUTAGE_END_MIN
        )


    try:

        result = run_function(
            **kwargs
        )


    finally:

        if old_outage_start is not None:

            outage_module.DUAL_OUTAGE_START_MIN = (
                old_outage_start
            )


        if old_outage_end is not None:

            outage_module.DUAL_OUTAGE_END_MIN = (
                old_outage_end
            )


        if (
            estimator_control[
                "kind"
            ]
            ==
            "module_global"
        ):

            setattr(
                outage_module,
                estimator_control[
                    "name"
                ],
                old_estimator_control,
            )


    result = dict(
        result
    )


    result[
        "experiment_estimator_age_min"
    ] = estimator_age_min


    result[
        "experiment_estimator_start_min"
    ] = estimator_start_min


    result[
        "experiment_outage_start_min"
    ] = (
        FIXED_OUTAGE_START_MIN
    )


    result[
        "experiment_outage_end_min"
    ] = (
        FIXED_OUTAGE_END_MIN
    )


    result[
        "experiment_outage_duration_min"
    ] = (
        FIXED_OUTAGE_DURATION_MIN
    )


    result[
        "fixed_initial_true_anomaly_deg"
    ] = (
        FIXED_INITIAL_TRUE_ANOMALY_DEG
    )


    result[
        "estimator_control_kind"
    ] = estimator_control[
        "kind"
    ]


    result[
        "estimator_control_name"
    ] = estimator_control[
        "name"
    ]


    result[
        "estimator_control_mode"
    ] = estimator_control[
        "mode"
    ]


    result[
        "estimator_control_value"
    ] = control_value


    return result


# ------------------------------------------------------------
# 7. PRE-FLIGHT SAFETY CHECK
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experiment 017-F"
)


print(
    "Controlled estimator-maturity intervention"
)


print(
    "========================================================================================================================"
)


print(
    f"Fixed initial true anomaly : "
    f"{FIXED_INITIAL_TRUE_ANOMALY_DEG:.1f} deg"
)


print(
    f"Fixed dual outage : "
    f"{FIXED_OUTAGE_START_MIN:.1f}-"
    f"{FIXED_OUTAGE_END_MIN:.1f} min"
)


print(
    f"Estimator ages : "
    f"{ESTIMATOR_AGES_MIN} min"
)


print(
    f"Replicates per age : "
    f"{NUMBER_OF_REPLICATES}"
)


print(
    f"Total expected missions : "
    f"{len(ESTIMATOR_AGES_MIN) * NUMBER_OF_REPLICATES}"
)


print()


if estimator_control is None:

    print(
        "ESTIMATOR CONTROL DISCOVERY : FAILED"
    )


    print()


    print(
        "017-F has NOT run any missions.\n"
        "The current simulation interface does not expose a "
        "recognized estimator-start / estimator-age control.\n\n"
        "We need one real estimator-control point before "
        "017-F can be scientifically valid."
    )


    raise RuntimeError(
        "017-F aborted before simulation: "
        "no estimator maturity control found."
    )


print(
    "ESTIMATOR CONTROL DISCOVERY : SUCCESS"
)


print(
    f"Control interface : "
    f"{estimator_control['kind']}"
)


print(
    f"Control name : "
    f"{estimator_control['name']}"
)


print(
    f"Control mode : "
    f"{estimator_control['mode']}"
)


print()


# ------------------------------------------------------------
# 8. DESIGN TABLE
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


    for estimator_age_min in (
        ESTIMATOR_AGES_MIN
    ):

        estimator_start_min = (
            FIXED_OUTAGE_START_MIN
            -
            estimator_age_min
        )


        design_rows.append(
            {
                "maturity_name":
                    (
                        f"AGE_"
                        f"{int(round(estimator_age_min)):02d}"
                        f"MIN"
                    ),

                "estimator_age_min":
                    float(
                        estimator_age_min
                    ),

                "estimator_start_min":
                    float(
                        estimator_start_min
                    ),

                "outage_start_min":
                    FIXED_OUTAGE_START_MIN,

                "outage_end_min":
                    FIXED_OUTAGE_END_MIN,

                "outage_duration_min":
                    FIXED_OUTAGE_DURATION_MIN,

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
        ESTIMATOR_AGES_MIN
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
# 9. CHECKPOINT / RESUME
# ------------------------------------------------------------

if results_path.exists():

    existing_results = pd.read_csv(
        results_path
    )


    required_checkpoint_columns = [
        "experiment_estimator_age_min",
        "replicate_index",
    ]


    if not all(
        column
        in existing_results.columns

        for column
        in required_checkpoint_columns
    ):

        raise RuntimeError(
            "Existing 017-F checkpoint does not "
            "contain the expected key columns."
        )


    existing_results = (
        existing_results
        .sort_values(
            [
                "replicate_index",
                "experiment_estimator_age_min",
            ]
        )
        .drop_duplicates(
            subset=[
                "experiment_estimator_age_min",
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
        index=False,
    )


    completed_keys = set(
        zip(
            existing_results[
                "experiment_estimator_age_min"
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

    existing_results = pd.DataFrame()

    completed_keys = set()


remaining_mask = []


for _, row in design.iterrows():

    key = (
        float(
            row[
                "estimator_age_min"
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
        "NO 017-F missions will be repeated."
    )


print()


# ------------------------------------------------------------
# 10. RUN ONLY MISSING MISSIONS
# ------------------------------------------------------------

new_rows = []


for local_index, (_, row) in enumerate(
    remaining_design.iterrows()
):

    estimator_age_min = float(
        row[
            "estimator_age_min"
        ]
    )


    estimator_start_min = float(
        row[
            "estimator_start_min"
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
        f"age="
        f"{estimator_age_min:5.1f} min | "
        f"start="
        f"{estimator_start_min:5.1f} min | "
        f"rep="
        f"{replicate_index:02d} | "
        f"seed="
        f"{scenario_seed}"
    )


    result = (
        run_controlled_maturity_scenario(
            estimator_age_min=
                estimator_age_min,

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
                "experiment_estimator_age_min",
            ]
        )
        .drop_duplicates(
            subset=[
                "experiment_estimator_age_min",
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
        index=False,
    )


    print(
        f"    sigma_start="
        f"{result['outage_start_sigma_m']:.4f} m | "
        f"RMSE="
        f"{result['outage_rmse_m']:.3f} m | "
        f"end="
        f"{result['outage_end_error_m']:.3f} m | "
        f"post="
        f"{result['post_recovery_rmse_m']:.3f} m | "
        f"NIS="
        f"{result['navigation_nis_mean']:.3f}"
    )


# ------------------------------------------------------------
# 11. RELOAD FINAL RESULTS
# ------------------------------------------------------------

if not results_path.exists():

    raise RuntimeError(
        "017-F results file does not exist."
    )


results = pd.read_csv(
    results_path
)


results = (
    results
    .sort_values(
        [
            "replicate_index",
            "experiment_estimator_age_min",
        ]
    )
    .drop_duplicates(
        subset=[
            "experiment_estimator_age_min",
            "replicate_index",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 12. CORE VALIDATION
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
            "experiment_estimator_age_min",
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


all_ages_complete = bool(
    np.all(
        results.groupby(
            "experiment_estimator_age_min"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)


all_replicates_cover_all_ages = bool(
    np.all(
        results.groupby(
            "replicate_index"
        )[
            "experiment_estimator_age_min"
        ]
        .nunique()
        ==
        len(
            ESTIMATOR_AGES_MIN
        )
    )
)


fixed_phase_valid = bool(
    np.allclose(
        results[
            "fixed_initial_true_anomaly_deg"
        ]
        .to_numpy(
            dtype=float
        ),

        FIXED_INITIAL_TRUE_ANOMALY_DEG,
    )
)


fixed_outage_valid = bool(
    np.allclose(
        results[
            "experiment_outage_start_min"
        ]
        .to_numpy(
            dtype=float
        ),

        FIXED_OUTAGE_START_MIN,
    )
    and
    np.allclose(
        results[
            "experiment_outage_end_min"
        ]
        .to_numpy(
            dtype=float
        ),

        FIXED_OUTAGE_END_MIN,
    )
)


age_start_identity_valid = bool(
    np.allclose(
        results[
            "experiment_estimator_age_min"
        ]
        .to_numpy(
            dtype=float
        )
        +
        results[
            "experiment_estimator_start_min"
        ]
        .to_numpy(
            dtype=float
        ),

        FIXED_OUTAGE_START_MIN,
    )
)


control_interface_consistent = bool(
    results[
        "estimator_control_kind"
    ].nunique()
    ==
    1
    and
    results[
        "estimator_control_name"
    ].nunique()
    ==
    1
    and
    results[
        "estimator_control_mode"
    ].nunique()
    ==
    1
)


core_columns = [
    "outage_start_sigma_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
    "post_recovery_rmse_m",
    "navigation_nis_mean",
    "navigation_nis_coverage_percent",
    "attitude_outage_rmse_deg",
    "attitude_end_outage_deg",
]


all_core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_columns
            ]
            .to_numpy(
                dtype=float
            )
        )
    )
)


if (
    "time_sync_error_s"
    in results.columns
):

    time_sync_valid = bool(
        np.max(
            np.abs(
                results[
                    "time_sync_error_s"
                ]
                .to_numpy(
                    dtype=float
                )
            )
        )
        <
        1.0e-12
    )


else:

    time_sync_valid = True


# ------------------------------------------------------------
# 13. MATURITY SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        "experiment_estimator_age_min",
        as_index=False,
    )
    .agg(
        runs=(
            "replicate_index",
            "size",
        ),

        estimator_start_mean_min=(
            "experiment_estimator_start_min",
            "mean",
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
# 14. FRIEDMAN TESTS
# ------------------------------------------------------------

FRIEDMAN_METRICS = [
    "outage_start_sigma_m",
    "outage_start_error_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "outage_end_sigma_m",
    "post_recovery_rmse_m",
]


friedman_rows = []


for metric in FRIEDMAN_METRICS:

    pivot = (
        results.pivot(
            index=
                "replicate_index",

            columns=
                "experiment_estimator_age_min",

            values=
                metric,
        )
        .reindex(
            columns=
                ESTIMATOR_AGES_MIN
        )
    )


    if pivot.isna().any().any():

        raise RuntimeError(
            "Incomplete repeated-measures "
            f"matrix for primary metric: {metric}"
        )


    age_samples = [
        pivot[
            age
        ].to_numpy(
            dtype=float
        )

        for age in ESTIMATOR_AGES_MIN
    ]


    statistic, p_value = (
        friedmanchisquare(
            *age_samples
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


    age_means = np.array(
        [
            np.mean(
                sample
            )

            for sample
            in age_samples
        ],
        dtype=float,
    )


    mean_range = float(
        np.max(
            age_means
        )
        -
        np.min(
            age_means
        )
    )


    overall_mean = float(
        np.mean(
            age_means
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

            "age_mean_range":
                mean_range,

            "relative_age_range_percent":
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
    index=False,
)


# ------------------------------------------------------------
# 15. SPEARMAN TREND TESTS
# ------------------------------------------------------------

CORRELATION_PAIRS = [
    (
        "experiment_estimator_age_min",
        "outage_start_sigma_m",
    ),

    (
        "experiment_estimator_age_min",
        "outage_rmse_m",
    ),

    (
        "experiment_estimator_age_min",
        "outage_end_error_m",
    ),

    (
        "outage_start_sigma_m",
        "outage_rmse_m",
    ),

    (
        "outage_start_sigma_m",
        "outage_end_error_m",
    ),

    (
        "pre_outage_pdop",
        "outage_rmse_m",
    ),
]


correlation_rows = []


for predictor, outcome in (
    CORRELATION_PAIRS
):

    x = (
        results[
            predictor
        ]
        .to_numpy(
            dtype=float
        )
    )


    y = (
        results[
            outcome
        ]
        .to_numpy(
            dtype=float
        )
    )


    finite_mask = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )


    rho, p_value = spearmanr(
        x[
            finite_mask
        ],

        y[
            finite_mask
        ],
    )


    correlation_rows.append(
        {
            "predictor":
                predictor,

            "outcome":
                outcome,

            "n":
                int(
                    np.sum(
                        finite_mask
                    )
                ),

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
    index=False,
)


# ------------------------------------------------------------
# 16. HOLM-CORRECTED POST-HOC WILCOXON
# ------------------------------------------------------------

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )


    m = len(
        p_values
    )


    order = np.argsort(
        p_values
    )


    adjusted = np.empty(
        m,
        dtype=float,
    )


    running_max = 0.0


    for rank_index, original_index in enumerate(
        order
    ):

        multiplier = (
            m
            -
            rank_index
        )


        candidate = min(
            1.0,

            multiplier
            *
            p_values[
                original_index
            ],
        )


        running_max = max(
            running_max,
            candidate,
        )


        adjusted[
            original_index
        ] = running_max


    return adjusted


posthoc_rows = []


outage_rmse_friedman_p = float(
    friedman_table.loc[
        friedman_table[
            "metric"
        ]
        ==
        "outage_rmse_m",

        "p_value",
    ]
    .iloc[
        0
    ]
)


if outage_rmse_friedman_p < 0.05:

    raw_p_values = []

    temporary_rows = []


    pivot_rmse = (
        results.pivot(
            index=
                "replicate_index",

            columns=
                "experiment_estimator_age_min",

            values=
                "outage_rmse_m",
        )
        .reindex(
            columns=
                ESTIMATOR_AGES_MIN
        )
    )


    for i in range(
        len(
            ESTIMATOR_AGES_MIN
        )
    ):

        for j in range(
            i + 1,
            len(
                ESTIMATOR_AGES_MIN
            )
        ):

            age_a = (
                ESTIMATOR_AGES_MIN[
                    i
                ]
            )


            age_b = (
                ESTIMATOR_AGES_MIN[
                    j
                ]
            )


            sample_a = (
                pivot_rmse[
                    age_a
                ]
                .to_numpy(
                    dtype=float
                )
            )


            sample_b = (
                pivot_rmse[
                    age_b
                ]
                .to_numpy(
                    dtype=float
                )
            )


            try:

                statistic, p_value = (
                    wilcoxon(
                        sample_a,
                        sample_b,

                        alternative=
                            "two-sided",

                        zero_method=
                            "wilcox",
                    )
                )


            except ValueError:

                statistic = 0.0

                p_value = 1.0


            raw_p_values.append(
                float(
                    p_value
                )
            )


            temporary_rows.append(
                {
                    "age_a_min":
                        float(
                            age_a
                        ),

                    "age_b_min":
                        float(
                            age_b
                        ),

                    "median_a_m":
                        float(
                            np.median(
                                sample_a
                            )
                        ),

                    "median_b_m":
                        float(
                            np.median(
                                sample_b
                            )
                        ),

                    "wilcoxon_statistic":
                        float(
                            statistic
                        ),

                    "p_raw":
                        float(
                            p_value
                        ),
                }
            )


    adjusted_p_values = (
        holm_adjust(
            raw_p_values
        )
    )


    for row, p_adjusted in zip(
        temporary_rows,
        adjusted_p_values,
    ):

        row[
            "p_holm"
        ] = float(
            p_adjusted
        )


        row[
            "significant_holm_0p05"
        ] = bool(
            p_adjusted
            <
            0.05
        )


        posthoc_rows.append(
            row
        )


posthoc_table = pd.DataFrame(
    posthoc_rows
)


posthoc_table.to_csv(
    posthoc_path,
    index=False,
)


# ------------------------------------------------------------
# 17. SCIENTIFIC FLAGS
# ------------------------------------------------------------

def get_corr(
    predictor,
    outcome,
):

    row = correlation_table[
        (
            correlation_table[
                "predictor"
            ]
            ==
            predictor
        )
        &
        (
            correlation_table[
                "outcome"
            ]
            ==
            outcome
        )
    ]


    if len(
        row
    ) == 0:

        return (
            np.nan,
            np.nan,
        )


    row = row.iloc[
        0
    ]


    return (
        float(
            row[
                "spearman_rho"
            ]
        ),

        float(
            row[
                "p_value"
            ]
        ),
    )


age_sigma_rho, age_sigma_p = (
    get_corr(
        "experiment_estimator_age_min",
        "outage_start_sigma_m",
    )
)


age_rmse_rho, age_rmse_p = (
    get_corr(
        "experiment_estimator_age_min",
        "outage_rmse_m",
    )
)


age_end_rho, age_end_p = (
    get_corr(
        "experiment_estimator_age_min",
        "outage_end_error_m",
    )
)


sigma_rmse_rho, sigma_rmse_p = (
    get_corr(
        "outage_start_sigma_m",
        "outage_rmse_m",
    )
)


sigma_end_rho, sigma_end_p = (
    get_corr(
        "outage_start_sigma_m",
        "outage_end_error_m",
    )
)


older_estimator_has_lower_sigma = bool(
    np.isfinite(
        age_sigma_p
    )
    and
    age_sigma_p
    <
    0.05
    and
    age_sigma_rho
    <
    0.0
)


older_estimator_has_lower_rmse = bool(
    np.isfinite(
        age_rmse_p
    )
    and
    age_rmse_p
    <
    0.05
    and
    age_rmse_rho
    <
    0.0
)


older_estimator_has_lower_end_error = bool(
    np.isfinite(
        age_end_p
    )
    and
    age_end_p
    <
    0.05
    and
    age_end_rho
    <
    0.0
)


sigma_predicts_rmse = bool(
    np.isfinite(
        sigma_rmse_p
    )
    and
    sigma_rmse_p
    <
    0.05
    and
    sigma_rmse_rho
    >
    0.0
)


sigma_predicts_end_error = bool(
    np.isfinite(
        sigma_end_p
    )
    and
    sigma_end_p
    <
    0.05
    and
    sigma_end_rho
    >
    0.0
)


maturity_intervention_supported = bool(
    older_estimator_has_lower_sigma
    and
    older_estimator_has_lower_rmse
    and
    older_estimator_has_lower_end_error
)


# ------------------------------------------------------------
# 18. GLOBAL VALIDATION
# ------------------------------------------------------------

validation_017f = bool(
    design_valid
    and
    all_runs_complete
    and
    unique_keys
    and
    all_ages_complete
    and
    all_replicates_cover_all_ages
    and
    fixed_phase_valid
    and
    fixed_outage_valid
    and
    age_start_identity_valid
    and
    control_interface_consistent
    and
    all_core_metrics_finite
    and
    time_sync_valid
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

            "all_ages_complete":
                all_ages_complete,

            "all_replicates_cover_all_ages":
                all_replicates_cover_all_ages,

            "fixed_phase_valid":
                fixed_phase_valid,

            "fixed_outage_valid":
                fixed_outage_valid,

            "age_start_identity_valid":
                age_start_identity_valid,

            "control_interface_consistent":
                control_interface_consistent,

            "all_core_metrics_finite":
                all_core_metrics_finite,

            "time_sync_valid":
                time_sync_valid,

            "global_validation":
                validation_017f,
        }
    ]
)


validation_table.to_csv(
    validation_path,
    index=False,
)


# ------------------------------------------------------------
# 19. PRINT RESULTS
# ------------------------------------------------------------

print()


print(
    "========================================================================================================================"
)


print(
    "AURORA — EXPERIMENT 017-F RESULTS"
)


print(
    "Controlled estimator-maturity intervention"
)


print(
    "========================================================================================================================"
)


print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)


print(
    f"Estimator control : "
    f"{estimator_control['kind']} / "
    f"{estimator_control['name']} / "
    f"{estimator_control['mode']}"
)


print(
    f"Fixed outage : "
    f"{FIXED_OUTAGE_START_MIN:.1f}-"
    f"{FIXED_OUTAGE_END_MIN:.1f} min"
)


print()


print(
    "----- MATURITY SUMMARY -----"
)


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


print(
    "----- REPEATED-MEASURES MATURITY TEST -----"
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

            "age_mean_range":
                lambda value:
                    f"{value:.4f}",

            "relative_age_range_percent":
                lambda value:
                    f"{value:.2f}",
        },
    )
)


print()


print(
    "----- MATURITY / ROBUSTNESS CORRELATIONS -----"
)


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


if len(
    posthoc_table
) > 0:

    print(
        "----- POST-HOC WILCOXON: OUTAGE RMSE -----"
    )


    print(
        posthoc_table.to_string(
            index=False,

            formatters={
                "median_a_m":
                    lambda value:
                        f"{value:.4f}",

                "median_b_m":
                    lambda value:
                        f"{value:.4f}",

                "wilcoxon_statistic":
                    lambda value:
                        f"{value:.3f}",

                "p_raw":
                    lambda value:
                        f"{value:.6f}",

                "p_holm":
                    lambda value:
                        f"{value:.6f}",
            },
        )
    )


    print()


print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Older estimator -> lower outage-start sigma : "
    f"{older_estimator_has_lower_sigma}"
)


print(
    f"Older estimator -> lower outage RMSE : "
    f"{older_estimator_has_lower_rmse}"
)


print(
    f"Older estimator -> lower end-of-outage error : "
    f"{older_estimator_has_lower_end_error}"
)


print(
    f"Outage-start sigma predicts outage RMSE : "
    f"{sigma_predicts_rmse}"
)


print(
    f"Outage-start sigma predicts end error : "
    f"{sigma_predicts_end_error}"
)


print(
    f"Controlled maturity mechanism supported : "
    f"{maturity_intervention_supported}"
)


print()


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
    f"All age/replicate keys unique : "
    f"{unique_keys}"
)


print(
    f"{NUMBER_OF_REPLICATES} replicates per age : "
    f"{all_ages_complete}"
)


print(
    f"Every replicate covers all maturity levels : "
    f"{all_replicates_cover_all_ages}"
)


print(
    f"Initial orbital phase fixed : "
    f"{fixed_phase_valid}"
)


print(
    f"Outage fixed at 60-75 min : "
    f"{fixed_outage_valid}"
)


print(
    f"Estimator age/start identity valid : "
    f"{age_start_identity_valid}"
)


print(
    f"Estimator-control interface consistent : "
    f"{control_interface_consistent}"
)


print(
    f"All required core metrics finite : "
    f"{all_core_metrics_finite}"
)


print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)


print()


print(
    f"VALIDATION GLOBALE 017-F : "
    f"{validation_017f}"
)


print(
    "========================================================================================================================"
)


# ============================================================
# 20. FIGURE — ESTIMATOR AGE VS START SIGMA
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_estimator_age_min"
    ],

    summary[
        "sigma_start_mean_m"
    ],

    yerr=
        summary[
            "sigma_start_std_m"
        ],

    marker="o",
)


plt.xlabel(
    "Estimator age at outage onset [min]"
)


plt.ylabel(
    "Position sigma at outage onset [m]"
)


plt.title(
    "AURORA — 017-F estimator maturity vs outage-start uncertainty"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017f_age_vs_start_sigma.png",

    dpi=200,
)


plt.show()


# ============================================================
# 21. FIGURE — ESTIMATOR AGE VS OUTAGE RMSE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_estimator_age_min"
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


plt.xlabel(
    "Estimator age at outage onset [min]"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-F controlled estimator maturity vs outage resilience"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017f_age_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 22. FIGURE — ESTIMATOR AGE VS END ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_estimator_age_min"
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


plt.xlabel(
    "Estimator age at outage onset [min]"
)


plt.ylabel(
    "Position error at end of outage [m]"
)


plt.title(
    "AURORA — 017-F estimator maturity vs end-of-outage error"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017f_age_vs_end_error.png",

    dpi=200,
)


plt.show()


# ============================================================
# 23. FIGURE — START SIGMA VS OUTAGE RMSE
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
    "AURORA — 017-F controlled uncertainty vs outage degradation"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017f_sigma_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 24. FIGURE — PAIRED REPLICATE TRAJECTORIES
# ============================================================

plt.figure(
    figsize=(10, 6)
)


for replicate_index in sorted(
    results[
        "replicate_index"
    ]
    .astype(
        int
    )
    .unique()
):

    replicate_data = (
        results[
            results[
                "replicate_index"
            ]
            ==
            replicate_index
        ]
        .sort_values(
            "experiment_estimator_age_min"
        )
    )


    plt.plot(
        replicate_data[
            "experiment_estimator_age_min"
        ],

        replicate_data[
            "outage_rmse_m"
        ],

        marker="o",
    )


plt.xlabel(
    "Estimator age at outage onset [min]"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-F paired replicate response to estimator maturity"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017f_paired_rmse_vs_age.png",

    dpi=200,
)


plt.show()