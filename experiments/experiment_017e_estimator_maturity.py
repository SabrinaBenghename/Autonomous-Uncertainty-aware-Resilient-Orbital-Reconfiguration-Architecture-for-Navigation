from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import (
    spearmanr,
    pearsonr,
    rankdata,
)


# ============================================================
# AURORA
# Experiment 017-E
#
# ESTIMATOR MATURITY / CONVERGENCE MECHANISM STUDY
#
#
# PURPOSE
# ------------------------------------------------------------
#
# Experiment 017-D established that:
#
#   - outage timing strongly changes outage performance
#   - outage_start_sigma_m strongly correlates with outage error
#   - PDOP does not explain the effect well
#
#
# 017-E now asks:
#
#   Is the timing effect consistent with estimator maturity /
#   convergence rather than instantaneous GNSS geometry?
#
#
# IMPORTANT
# ------------------------------------------------------------
#
# NO NEW MISSIONS ARE RUN.
#
# This experiment reuses the already validated 64 missions
# from Experiment 017-D.
#
#
# Analyses:
#
#   1. Estimator uncertainty vs mission time
#   2. Exponential convergence model for sigma
#   3. Sigma vs outage RMSE
#   4. Sigma vs end-of-outage error
#   5. Partial Spearman correlation controlling for time
#   6. Partial Spearman controlling for time + PDOP
#   7. Within-timing residual analysis
#   8. Comparison of explanatory strength:
#
#          mission time
#          PDOP
#          visible satellites
#          outage-start sigma
#
#
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
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


input_path = (
    table_directory
    /
    "phase17_017d_dual_outage_timing_sweep.csv"
)


summary_path = (
    table_directory
    /
    "phase17_017e_estimator_maturity_summary.csv"
)


correlation_path = (
    table_directory
    /
    "phase17_017e_mechanism_correlations.csv"
)


partial_path = (
    table_directory
    /
    "phase17_017e_partial_correlations.csv"
)


convergence_path = (
    table_directory
    /
    "phase17_017e_convergence_model.csv"
)


validation_path = (
    table_directory
    /
    "phase17_017e_validation.csv"
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


table_directory.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD 017-D
# ------------------------------------------------------------

if not input_path.exists():

    raise RuntimeError(
        "\n017-E cannot start because the validated "
        "017-D results file was not found:\n"
        f"{input_path}\n"
        "\n017-E does NOT rerun missions. "
        "It requires the existing 017-D CSV."
    )


results = pd.read_csv(
    input_path
)


# ------------------------------------------------------------
# 3. REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "experiment_outage_start_min",
    "experiment_outage_end_min",
    "replicate_index",
    "scenario_seed",
    "outage_start_sigma_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "pre_outage_pdop",
    "natural_visible_satellites_mean",
    "navigation_nis_mean",
    "attitude_outage_rmse_deg",
]


missing_columns = [
    column
    for column in required_columns
    if column not in results.columns
]


if len(
    missing_columns
) > 0:

    raise RuntimeError(
        "\n017-E missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ------------------------------------------------------------
# 4. CLEAN / SORT
# ------------------------------------------------------------

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


EXPECTED_TIMINGS = [
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
]


EXPECTED_REPLICATES = 8


EXPECTED_RUNS = (
    len(
        EXPECTED_TIMINGS
    )
    *
    EXPECTED_REPLICATES
)


# ------------------------------------------------------------
# 5. DATA VALIDATION
# ------------------------------------------------------------

run_count_valid = bool(
    len(
        results
    )
    ==
    EXPECTED_RUNS
)


unique_keys_valid = bool(
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
    EXPECTED_RUNS
)


timings_present = sorted(
    results[
        "experiment_outage_start_min"
    ]
    .astype(
        float
    )
    .unique()
    .tolist()
)


timings_valid = bool(
    np.allclose(
        timings_present,
        EXPECTED_TIMINGS,
    )
)


replicates_per_timing_valid = bool(
    np.all(
        results.groupby(
            "experiment_outage_start_min"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        EXPECTED_REPLICATES
    )
)


primary_columns = [
    "experiment_outage_start_min",
    "outage_start_sigma_m",
    "outage_rmse_m",
    "outage_end_error_m",
    "pre_outage_pdop",
    "natural_visible_satellites_mean",
]


primary_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                primary_columns
            ].to_numpy(
                dtype=float
            )
        )
    )
)


# ------------------------------------------------------------
# 6. SUMMARY BY OUTAGE START TIME
#
# outage_start_sigma_m is measured immediately before the
# outage and is therefore used as the estimator-maturity
# indicator.
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

        sigma_start_mean_m=(
            "outage_start_sigma_m",
            "mean",
        ),

        sigma_start_std_m=(
            "outage_start_sigma_m",
            "std",
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

        pdop_mean=(
            "pre_outage_pdop",
            "mean",
        ),

        pdop_std=(
            "pre_outage_pdop",
            "std",
        ),

        visible_satellites_mean=(
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
    index=False
)


# ------------------------------------------------------------
# 7. SIMPLE SPEARMAN CORRELATIONS
# ------------------------------------------------------------

CORRELATION_PAIRS = [
    (
        "experiment_outage_start_min",
        "outage_start_sigma_m",
        "Mission time",
        "Estimator sigma",
    ),

    (
        "experiment_outage_start_min",
        "outage_rmse_m",
        "Mission time",
        "Outage RMSE",
    ),

    (
        "experiment_outage_start_min",
        "outage_end_error_m",
        "Mission time",
        "End-of-outage error",
    ),

    (
        "outage_start_sigma_m",
        "outage_rmse_m",
        "Estimator sigma",
        "Outage RMSE",
    ),

    (
        "outage_start_sigma_m",
        "outage_end_error_m",
        "Estimator sigma",
        "End-of-outage error",
    ),

    (
        "pre_outage_pdop",
        "outage_rmse_m",
        "PDOP",
        "Outage RMSE",
    ),

    (
        "natural_visible_satellites_mean",
        "outage_rmse_m",
        "Visible satellites",
        "Outage RMSE",
    ),
]


correlation_rows = []


for (
    predictor,
    outcome,
    predictor_label,
    outcome_label,
) in CORRELATION_PAIRS:

    x = results[
        predictor
    ].to_numpy(
        dtype=float
    )


    y = results[
        outcome
    ].to_numpy(
        dtype=float
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


    x_valid = x[
        finite_mask
    ]


    y_valid = y[
        finite_mask
    ]


    if len(
        x_valid
    ) >= 3:

        rho, p_value = spearmanr(
            x_valid,
            y_valid,
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

            "predictor_label":
                predictor_label,

            "outcome_label":
                outcome_label,

            "n":
                int(
                    len(
                        x_valid
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
# 8. PARTIAL SPEARMAN FUNCTION
#
# Spearman correlation is Pearson correlation on ranks.
#
# For partial Spearman:
#
#   1. Rank X
#   2. Rank Y
#   3. Rank controls
#   4. Regress X and Y separately against controls
#   5. Correlate residuals
#
# This tests whether sigma still predicts outage performance
# after statistically removing mission-time effects.
# ------------------------------------------------------------

def partial_spearman(
    dataframe,
    predictor,
    outcome,
    controls,
):

    used_columns = [
        predictor,
        outcome,
    ] + list(
        controls
    )


    data = (
        dataframe[
            used_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
        .copy()
    )


    n = len(
        data
    )


    if n < 5:

        return {
            "n":
                n,

            "rho":
                np.nan,

            "p_value":
                np.nan,

            "condition_number":
                np.nan,
        }


    x_rank = rankdata(
        data[
            predictor
        ].to_numpy(
            dtype=float
        )
    )


    y_rank = rankdata(
        data[
            outcome
        ].to_numpy(
            dtype=float
        )
    )


    control_rank_columns = []


    for control in controls:

        control_rank_columns.append(
            rankdata(
                data[
                    control
                ].to_numpy(
                    dtype=float
                )
            )
        )


    controls_matrix = np.column_stack(
        control_rank_columns
    )


    design_matrix = np.column_stack(
        [
            np.ones(
                n
            ),
            controls_matrix,
        ]
    )


    beta_x = np.linalg.lstsq(
        design_matrix,
        x_rank,
        rcond=None,
    )[0]


    beta_y = np.linalg.lstsq(
        design_matrix,
        y_rank,
        rcond=None,
    )[0]


    x_residual = (
        x_rank
        -
        design_matrix
        @
        beta_x
    )


    y_residual = (
        y_rank
        -
        design_matrix
        @
        beta_y
    )


    if (
        np.std(
            x_residual
        )
        <
        1.0e-12
        or
        np.std(
            y_residual
        )
        <
        1.0e-12
    ):

        rho = np.nan
        p_value = np.nan


    else:

        rho, p_value = pearsonr(
            x_residual,
            y_residual,
        )


    condition_number = float(
        np.linalg.cond(
            design_matrix
        )
    )


    return {
        "n":
            int(
                n
            ),

        "rho":
            float(
                rho
            ),

        "p_value":
            float(
                p_value
            ),

        "condition_number":
            condition_number,
    }


# ------------------------------------------------------------
# 9. PARTIAL CORRELATION TESTS
# ------------------------------------------------------------

partial_tests = [
    {
        "name":
            "sigma_vs_rmse_control_time",

        "predictor":
            "outage_start_sigma_m",

        "outcome":
            "outage_rmse_m",

        "controls":
            [
                "experiment_outage_start_min"
            ],
    },

    {
        "name":
            "sigma_vs_end_error_control_time",

        "predictor":
            "outage_start_sigma_m",

        "outcome":
            "outage_end_error_m",

        "controls":
            [
                "experiment_outage_start_min"
            ],
    },

    {
        "name":
            "sigma_vs_rmse_control_time_pdop",

        "predictor":
            "outage_start_sigma_m",

        "outcome":
            "outage_rmse_m",

        "controls":
            [
                "experiment_outage_start_min",
                "pre_outage_pdop",
            ],
    },

    {
        "name":
            "sigma_vs_end_error_control_time_pdop",

        "predictor":
            "outage_start_sigma_m",

        "outcome":
            "outage_end_error_m",

        "controls":
            [
                "experiment_outage_start_min",
                "pre_outage_pdop",
            ],
    },

    {
        "name":
            "pdop_vs_rmse_control_time",

        "predictor":
            "pre_outage_pdop",

        "outcome":
            "outage_rmse_m",

        "controls":
            [
                "experiment_outage_start_min"
            ],
    },
]


partial_rows = []


for test in partial_tests:

    partial_result = partial_spearman(
        dataframe=
            results,

        predictor=
            test[
                "predictor"
            ],

        outcome=
            test[
                "outcome"
            ],

        controls=
            test[
                "controls"
            ],
    )


    partial_rows.append(
        {
            "test":
                test[
                    "name"
                ],

            "predictor":
                test[
                    "predictor"
                ],

            "outcome":
                test[
                    "outcome"
                ],

            "controls":
                ",".join(
                    test[
                        "controls"
                    ]
                ),

            "n":
                partial_result[
                    "n"
                ],

            "partial_spearman_rho":
                partial_result[
                    "rho"
                ],

            "p_value":
                partial_result[
                    "p_value"
                ],

            "condition_number":
                partial_result[
                    "condition_number"
                ],

            "significant_0p05":
                bool(
                    np.isfinite(
                        partial_result[
                            "p_value"
                        ]
                    )
                    and
                    partial_result[
                        "p_value"
                    ]
                    <
                    0.05
                ),
        }
    )


partial_table = pd.DataFrame(
    partial_rows
)


partial_table.to_csv(
    partial_path,
    index=False
)


# ------------------------------------------------------------
# 10. WITHIN-TIMING RESIDUAL ANALYSIS
#
# Mission time and estimator sigma are strongly coupled.
#
# To ask whether run-to-run deviations in sigma inside the
# SAME outage timing also predict run-to-run degradation,
# remove each timing's group mean.
#
# This is useful but must be interpreted carefully if sigma
# has almost no within-timing stochastic variation.
# ------------------------------------------------------------

results[
    "sigma_within_timing_residual"
] = (
    results[
        "outage_start_sigma_m"
    ]
    -
    results.groupby(
        "experiment_outage_start_min"
    )[
        "outage_start_sigma_m"
    ]
    .transform(
        "mean"
    )
)


results[
    "rmse_within_timing_residual"
] = (
    results[
        "outage_rmse_m"
    ]
    -
    results.groupby(
        "experiment_outage_start_min"
    )[
        "outage_rmse_m"
    ]
    .transform(
        "mean"
    )
)


results[
    "end_error_within_timing_residual"
] = (
    results[
        "outage_end_error_m"
    ]
    -
    results.groupby(
        "experiment_outage_start_min"
    )[
        "outage_end_error_m"
    ]
    .transform(
        "mean"
    )
)


sigma_total_std = float(
    np.std(
        results[
            "outage_start_sigma_m"
        ].to_numpy(
            dtype=float
        ),
        ddof=1,
    )
)


sigma_within_std = float(
    np.std(
        results[
            "sigma_within_timing_residual"
        ].to_numpy(
            dtype=float
        ),
        ddof=1,
    )
)


within_sigma_fraction = float(
    sigma_within_std
    /
    max(
        sigma_total_std,
        1.0e-12,
    )
)


within_sigma_variation_sufficient = bool(
    within_sigma_fraction
    >=
    0.05
)


if sigma_within_std > 1.0e-12:

    within_sigma_rmse_rho, (
        within_sigma_rmse_p
    ) = spearmanr(
        results[
            "sigma_within_timing_residual"
        ],

        results[
            "rmse_within_timing_residual"
        ],
    )


    within_sigma_end_rho, (
        within_sigma_end_p
    ) = spearmanr(
        results[
            "sigma_within_timing_residual"
        ],

        results[
            "end_error_within_timing_residual"
        ],
    )


else:

    within_sigma_rmse_rho = np.nan
    within_sigma_rmse_p = np.nan

    within_sigma_end_rho = np.nan
    within_sigma_end_p = np.nan


# ------------------------------------------------------------
# 11. ESTIMATOR CONVERGENCE MODEL
#
# Model:
#
#     sigma(t) = sigma_floor + A * exp(-t / tau)
#
# tau:
#     approximate estimator convergence time constant
#
# This is descriptive rather than a proof of filter dynamics.
# ------------------------------------------------------------

def exponential_convergence_model(
    time_min,
    sigma_floor,
    amplitude,
    tau_min,
):

    return (
        sigma_floor
        +
        amplitude
        *
        np.exp(
            -
            time_min
            /
            tau_min
        )
    )


time_values = summary[
    "experiment_outage_start_min"
].to_numpy(
    dtype=float
)


sigma_mean_values = summary[
    "sigma_start_mean_m"
].to_numpy(
    dtype=float
)


initial_floor_guess = float(
    np.min(
        sigma_mean_values
    )
)


initial_amplitude_guess = float(
    np.max(
        sigma_mean_values
    )
    -
    initial_floor_guess
)


initial_tau_guess = 30.0


convergence_fit_valid = False

sigma_floor = np.nan
amplitude = np.nan
tau_min = np.nan
fit_r_squared = np.nan
time_to_10_percent_of_initial_excess_min = np.nan


try:

    parameters, covariance = curve_fit(
        exponential_convergence_model,

        time_values,

        sigma_mean_values,

        p0=[
            initial_floor_guess,
            initial_amplitude_guess,
            initial_tau_guess,
        ],

        bounds=(
            [
                0.0,
                0.0,
                1.0,
            ],

            [
                10.0,
                10.0,
                1000.0,
            ],
        ),

        maxfev=50_000,
    )


    (
        sigma_floor,
        amplitude,
        tau_min,
    ) = parameters


    predicted_sigma = (
        exponential_convergence_model(
            time_values,
            sigma_floor,
            amplitude,
            tau_min,
        )
    )


    residual_sum_squares = float(
        np.sum(
            (
                sigma_mean_values
                -
                predicted_sigma
            )
            **
            2
        )
    )


    total_sum_squares = float(
        np.sum(
            (
                sigma_mean_values
                -
                np.mean(
                    sigma_mean_values
                )
            )
            **
            2
        )
    )


    fit_r_squared = float(
        1.0
        -
        residual_sum_squares
        /
        max(
            total_sum_squares,
            1.0e-12,
        )
    )


    # Time for the exponential excess uncertainty
    # to fall to 10% of its t=0 value:
    #
    # exp(-t/tau) = 0.10

    time_to_10_percent_of_initial_excess_min = float(
        tau_min
        *
        np.log(
            10.0
        )
    )


    convergence_fit_valid = True


except Exception as exception:

    print(
        "\nWARNING: convergence fit failed:"
    )

    print(
        exception
    )


convergence_table = pd.DataFrame(
    [
        {
            "model":
                (
                    "sigma(t) = sigma_floor "
                    "+ amplitude * exp(-t/tau)"
                ),

            "sigma_floor_m":
                float(
                    sigma_floor
                ),

            "amplitude_m":
                float(
                    amplitude
                ),

            "tau_min":
                float(
                    tau_min
                ),

            "r_squared":
                float(
                    fit_r_squared
                ),

            "time_to_10_percent_excess_min":
                float(
                    time_to_10_percent_of_initial_excess_min
                ),

            "fit_valid":
                convergence_fit_valid,
        }
    ]
)


convergence_table.to_csv(
    convergence_path,
    index=False
)


# ------------------------------------------------------------
# 12. EXTRACT IMPORTANT RESULTS
# ------------------------------------------------------------

def get_simple_correlation(
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


time_sigma_rho, time_sigma_p = (
    get_simple_correlation(
        "experiment_outage_start_min",
        "outage_start_sigma_m",
    )
)


time_rmse_rho, time_rmse_p = (
    get_simple_correlation(
        "experiment_outage_start_min",
        "outage_rmse_m",
    )
)


sigma_rmse_rho, sigma_rmse_p = (
    get_simple_correlation(
        "outage_start_sigma_m",
        "outage_rmse_m",
    )
)


sigma_end_rho, sigma_end_p = (
    get_simple_correlation(
        "outage_start_sigma_m",
        "outage_end_error_m",
    )
)


pdop_rmse_rho, pdop_rmse_p = (
    get_simple_correlation(
        "pre_outage_pdop",
        "outage_rmse_m",
    )
)


# ------------------------------------------------------------
# 13. SCIENTIFIC FLAGS
# ------------------------------------------------------------

estimator_uncertainty_decreases_with_time = bool(
    np.isfinite(
        time_sigma_p
    )
    and
    time_sigma_p
    <
    0.05
    and
    time_sigma_rho
    <
    0.0
)


outage_performance_improves_with_time = bool(
    np.isfinite(
        time_rmse_p
    )
    and
    time_rmse_p
    <
    0.05
    and
    time_rmse_rho
    <
    0.0
)


sigma_predicts_outage_rmse = bool(
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


pdop_predicts_outage_rmse = bool(
    np.isfinite(
        pdop_rmse_p
    )
    and
    pdop_rmse_p
    <
    0.05
)


mechanism_evidence_consistent = bool(
    estimator_uncertainty_decreases_with_time
    and
    outage_performance_improves_with_time
    and
    sigma_predicts_outage_rmse
    and
    sigma_predicts_end_error
)


# ------------------------------------------------------------
# 14. GLOBAL VALIDATION
# ------------------------------------------------------------

validation_017e = bool(
    run_count_valid
    and
    unique_keys_valid
    and
    timings_valid
    and
    replicates_per_timing_valid
    and
    primary_metrics_finite
)


validation_table = pd.DataFrame(
    [
        {
            "run_count_valid":
                run_count_valid,

            "unique_keys_valid":
                unique_keys_valid,

            "timings_valid":
                timings_valid,

            "replicates_per_timing_valid":
                replicates_per_timing_valid,

            "primary_metrics_finite":
                primary_metrics_finite,

            "convergence_fit_valid":
                convergence_fit_valid,

            "global_validation":
                validation_017e,
        }
    ]
)


validation_table.to_csv(
    validation_path,
    index=False
)


# ------------------------------------------------------------
# 15. PRINT HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — EXPERIMENT 017-E"
)


print(
    "Estimator maturity / convergence mechanism study"
)


print(
    "========================================================================================================================"
)


print(
    "Simulation mode : EXISTING 017-D DATA ONLY"
)


print(
    "New missions executed : 0"
)


print(
    f"017-D missions loaded : "
    f"{len(results)}"
)


print()


# ------------------------------------------------------------
# 16. MATURITY SUMMARY
# ------------------------------------------------------------

print(
    "----- ESTIMATOR MATURITY SUMMARY -----"
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


# ------------------------------------------------------------
# 17. SIMPLE CORRELATIONS
# ------------------------------------------------------------

print(
    "----- SIMPLE SPEARMAN CORRELATIONS -----"
)


print(
    correlation_table[
        [
            "predictor_label",
            "outcome_label",
            "n",
            "spearman_rho",
            "p_value",
            "significant_0p05",
        ]
    ]
    .to_string(
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


# ------------------------------------------------------------
# 18. PARTIAL CORRELATIONS
# ------------------------------------------------------------

print(
    "----- PARTIAL SPEARMAN CORRELATIONS -----"
)


print(
    partial_table.to_string(
        index=False,

        formatters={
            "partial_spearman_rho":
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

            "condition_number":
                lambda value:
                    (
                        f"{value:.2f}"
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
# 19. WITHIN-TIMING ANALYSIS
# ------------------------------------------------------------

print(
    "----- WITHIN-TIMING RESIDUAL ANALYSIS -----"
)


print(
    f"Total sigma standard deviation : "
    f"{sigma_total_std:.6f} m"
)


print(
    f"Within-timing sigma standard deviation : "
    f"{sigma_within_std:.6f} m"
)


print(
    f"Within/total sigma variation fraction : "
    f"{within_sigma_fraction:.4f}"
)


print(
    f"Sufficient within-timing sigma variation : "
    f"{within_sigma_variation_sufficient}"
)


print(
    f"Within-timing sigma vs RMSE rho : "
    f"{within_sigma_rmse_rho:.4f}"
)


print(
    f"Within-timing sigma vs RMSE p : "
    f"{within_sigma_rmse_p:.6f}"
)


print(
    f"Within-timing sigma vs end-error rho : "
    f"{within_sigma_end_rho:.4f}"
)


print(
    f"Within-timing sigma vs end-error p : "
    f"{within_sigma_end_p:.6f}"
)


print()


# ------------------------------------------------------------
# 20. CONVERGENCE MODEL
# ------------------------------------------------------------

print(
    "----- ESTIMATOR CONVERGENCE MODEL -----"
)


if convergence_fit_valid:

    print(
        "Model : "
        "sigma(t) = sigma_floor + A exp(-t/tau)"
    )


    print(
        f"Estimated uncertainty floor : "
        f"{sigma_floor:.4f} m"
    )


    print(
        f"Estimated exponential amplitude : "
        f"{amplitude:.4f} m"
    )


    print(
        f"Estimated convergence time constant tau : "
        f"{tau_min:.2f} min"
    )


    print(
        f"Model R² : "
        f"{fit_r_squared:.4f}"
    )


    print(
        f"Time to 10% of initial excess uncertainty : "
        f"{time_to_10_percent_of_initial_excess_min:.2f} min"
    )


else:

    print(
        "Convergence model fit : FAILED"
    )


print()


# ------------------------------------------------------------
# 21. SCIENTIFIC INTERPRETATION FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Estimator uncertainty significantly decreases "
    f"with mission time : "
    f"{estimator_uncertainty_decreases_with_time}"
)


print(
    f"Outage performance significantly improves "
    f"with mission time : "
    f"{outage_performance_improves_with_time}"
)


print(
    f"Outage-start sigma significantly predicts "
    f"outage RMSE : "
    f"{sigma_predicts_outage_rmse}"
)


print(
    f"Outage-start sigma significantly predicts "
    f"end-of-outage error : "
    f"{sigma_predicts_end_error}"
)


print(
    f"PDOP significantly predicts outage RMSE : "
    f"{pdop_predicts_outage_rmse}"
)


print(
    f"Overall evidence consistent with estimator-"
    f"maturity mechanism : "
    f"{mechanism_evidence_consistent}"
)


print()


# ------------------------------------------------------------
# 22. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"64 017-D missions loaded : "
    f"{run_count_valid}"
)


print(
    f"All timing/replicate keys unique : "
    f"{unique_keys_valid}"
)


print(
    f"All expected outage timings present : "
    f"{timings_valid}"
)


print(
    f"8 replicates per timing : "
    f"{replicates_per_timing_valid}"
)


print(
    f"All primary mechanism metrics finite : "
    f"{primary_metrics_finite}"
)


print(
    f"Convergence model fit valid : "
    f"{convergence_fit_valid}"
)


print()


print(
    f"VALIDATION GLOBALE 017-E : "
    f"{validation_017e}"
)


print(
    "========================================================================================================================"
)


# ============================================================
# 23. FIGURE — ESTIMATOR SIGMA VS MISSION TIME
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_outage_start_min"
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


if convergence_fit_valid:

    smooth_time = np.linspace(
        0.0,
        135.0,
        500,
    )


    smooth_sigma = (
        exponential_convergence_model(
            smooth_time,
            sigma_floor,
            amplitude,
            tau_min,
        )
    )


    plt.plot(
        smooth_time,
        smooth_sigma,
    )


plt.xlabel(
    "Mission time at outage onset [min]"
)


plt.ylabel(
    "Position sigma at outage onset [m]"
)


plt.title(
    "AURORA — 017-E estimator uncertainty convergence"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_sigma_convergence.png",

    dpi=200,
)


plt.show()


# ============================================================
# 24. FIGURE — SIGMA VS OUTAGE RMSE
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
    "AURORA — 017-E estimator uncertainty vs outage RMSE"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_sigma_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 25. FIGURE — SIGMA VS END-OF-OUTAGE ERROR
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    results[
        "outage_start_sigma_m"
    ],

    results[
        "outage_end_error_m"
    ],
)


plt.xlabel(
    "Position sigma at outage onset [m]"
)


plt.ylabel(
    "Position error at end of outage [m]"
)


plt.title(
    "AURORA — 017-E estimator uncertainty vs end-of-outage error"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_sigma_vs_end_error.png",

    dpi=200,
)


plt.show()


# ============================================================
# 26. FIGURE — TIME VS OUTAGE RMSE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    summary[
        "experiment_outage_start_min"
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
    "Mission time at outage onset [min]"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-E outage vulnerability vs estimator maturity"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_outage_rmse_vs_time.png",

    dpi=200,
)


plt.show()


# ============================================================
# 27. FIGURE — PDOP VS OUTAGE RMSE
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
    "PDOP before outage"
)


plt.ylabel(
    "Dual-outage RMSE [m]"
)


plt.title(
    "AURORA — 017-E PDOP vs outage RMSE"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_pdop_vs_outage_rmse.png",

    dpi=200,
)


plt.show()


# ============================================================
# 28. FIGURE — NORMALIZED MECHANISM COMPARISON
#
# Shows whether estimator uncertainty and outage vulnerability
# evolve together over mission time.
# ============================================================

sigma_normalized = (
    summary[
        "sigma_start_mean_m"
    ]
    /
    np.max(
        summary[
            "sigma_start_mean_m"
        ]
    )
)


rmse_normalized = (
    summary[
        "outage_rmse_mean_m"
    ]
    /
    np.max(
        summary[
            "outage_rmse_mean_m"
        ]
    )
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    summary[
        "experiment_outage_start_min"
    ],

    sigma_normalized,

    marker="o",

    label=
        "Normalized estimator sigma",
)


plt.plot(
    summary[
        "experiment_outage_start_min"
    ],

    rmse_normalized,

    marker="o",

    label=
        "Normalized outage RMSE",
)


plt.xlabel(
    "Mission time at outage onset [min]"
)


plt.ylabel(
    "Normalized value"
)


plt.title(
    "AURORA — 017-E estimator maturity and outage vulnerability"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017e_maturity_vs_vulnerability.png",

    dpi=200,
)


plt.show()