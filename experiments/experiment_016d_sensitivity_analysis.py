from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.validation.sensitivity_analysis import (
    PHASE16_INPUT_PARAMETERS,
    DERIVED_SENSITIVITY_PREDICTORS,
    PARAMETER_SUBSYSTEM,
    add_derived_sensitivity_predictors,
    compute_spearman_screening,
    compute_standardized_ridge_screening,
    build_top_factor_table
)


# ============================================================
# AURORA
# Experience 016-D
#
# MULTIDIMENSIONAL SENSITIVITY SCREENING
#
#
# Inputs:
#
#       016-A
#           64-scenario Latin Hypercube
#
#       016-B
#           navigation / integrity robustness
#
#       016-C
#           frozen AI / FDIR robustness
#
#
# Primary method:
#
#       Spearman rank correlations
#       + Benjamini-Hochberg FDR correction
#
#
# Secondary method:
#
#       standardized multivariate Ridge regression
#
#
# Objective:
#
# Determine which sampled uncertainties are most strongly
# associated with:
#
#       navigation accuracy
#       integrity benefit
#       estimator consistency
#       FDIR detection
#       hybrid AI detection
#       hybrid false alarms
#
#
# IMPORTANT:
#
# This is sensitivity SCREENING.
#
# It does not establish causality or Sobol sensitivity indices.
# ============================================================


# ------------------------------------------------------------
# 1. INPUT FILES
# ------------------------------------------------------------

campaign_path = (
    Path("data")
    /
    "phase16"
    /
    "robustness_campaign_016a.csv"
)


navigation_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016b_navigation_robustness.csv"
)


ai_path = (
    Path("results")
    /
    "tables"
    /
    "phase16_016c_ai_fdir_robustness.csv"
)


for required_path in [
    campaign_path,
    navigation_path,
    ai_path
]:

    if not required_path.exists():

        raise FileNotFoundError(
            f"Required Phase-16 file missing: "
            f"{required_path}"
        )


# ------------------------------------------------------------
# 2. OUTPUT FILES
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


table_directory.mkdir(
    parents=True,
    exist_ok=True
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


analysis_dataset_path = (
    table_directory
    /
    "phase16_016d_analysis_dataset.csv"
)


spearman_path = (
    table_directory
    /
    "phase16_016d_spearman_sensitivity.csv"
)


derived_path = (
    table_directory
    /
    "phase16_016d_derived_sensitivity.csv"
)


ridge_path = (
    table_directory
    /
    "phase16_016d_ridge_sensitivity.csv"
)


ridge_summary_path = (
    table_directory
    /
    "phase16_016d_ridge_model_summary.csv"
)


top_factors_path = (
    table_directory
    /
    "phase16_016d_top_factors.csv"
)


# ------------------------------------------------------------
# 3. LOAD
# ------------------------------------------------------------

campaign = pd.read_csv(
    campaign_path
)


navigation = pd.read_csv(
    navigation_path
)


ai_results = pd.read_csv(
    ai_path
)


# ------------------------------------------------------------
# 4. OUTPUT METRICS
# ------------------------------------------------------------

NAVIGATION_OUTPUTS = [

    "protected_mission_rmse",

    "protected_low_redundancy_rmse",

    "protected_dual_outage_rmse",

    "protected_nis_mean",

    "gain_mission_rmse_m",

    "gain_dual_rmse_m"
]


AI_OUTPUTS = [

    "fdir_pd",

    "hybrid_pd",

    "hybrid_pfa",

    "hybrid_minus_fdir_pd"
]


OUTPUT_METRICS = (
    NAVIGATION_OUTPUTS
    +
    AI_OUTPUTS
)


# ------------------------------------------------------------
# 5. VERIFY REQUIRED COLUMNS
# ------------------------------------------------------------

required_campaign_columns = (
    [
        "scenario_id"
    ]
    +
    PHASE16_INPUT_PARAMETERS
)


required_navigation_columns = (
    [
        "scenario_id"
    ]
    +
    NAVIGATION_OUTPUTS
)


required_ai_columns = (
    [
        "scenario_id"
    ]
    +
    AI_OUTPUTS
)


def missing_columns(
    dataframe,
    required_columns
):

    return [
        column
        for column in required_columns
        if column
        not in dataframe.columns
    ]


missing_campaign = missing_columns(
    campaign,
    required_campaign_columns
)


missing_navigation = missing_columns(
    navigation,
    required_navigation_columns
)


missing_ai = missing_columns(
    ai_results,
    required_ai_columns
)


if len(
    missing_campaign
) > 0:

    raise RuntimeError(
        "Missing campaign columns: "
        +
        str(
            missing_campaign
        )
    )


if len(
    missing_navigation
) > 0:

    raise RuntimeError(
        "Missing navigation columns: "
        +
        str(
            missing_navigation
        )
    )


if len(
    missing_ai
) > 0:

    raise RuntimeError(
        "Missing AI columns: "
        +
        str(
            missing_ai
        )
    )


# ------------------------------------------------------------
# 6. SCENARIO-ID VALIDATION BEFORE MERGE
# ------------------------------------------------------------

campaign_ids = set(
    campaign[
        "scenario_id"
    ].astype(
        str
    )
)


navigation_ids = set(
    navigation[
        "scenario_id"
    ].astype(
        str
    )
)


ai_ids = set(
    ai_results[
        "scenario_id"
    ].astype(
        str
    )
)


scenario_sets_identical = (
    campaign_ids
    ==
    navigation_ids
    ==
    ai_ids
)


if not scenario_sets_identical:

    raise RuntimeError(
        "016-A/016-B/016-C scenario IDs do not match."
    )


# ------------------------------------------------------------
# 7. MERGE INTO ONE ANALYSIS TABLE
# ------------------------------------------------------------

analysis = (
    campaign[
        required_campaign_columns
    ]
    .merge(
        navigation[
            required_navigation_columns
        ],

        on=
            "scenario_id",

        how=
            "inner",

        validate=
            "one_to_one"
    )
    .merge(
        ai_results[
            required_ai_columns
        ],

        on=
            "scenario_id",

        how=
            "inner",

        validate=
            "one_to_one"
    )
)


analysis = (
    add_derived_sensitivity_predictors(
        analysis
    )
)


analysis.to_csv(
    analysis_dataset_path,
    index=False
)


# ------------------------------------------------------------
# 8. BASIC DATA VALIDATION
# ------------------------------------------------------------

correct_sample_count = (
    len(
        analysis
    )
    ==
    64
)


all_inputs_finite = bool(
    np.all(
        np.isfinite(
            analysis[
                PHASE16_INPUT_PARAMETERS
            ].to_numpy(
                dtype=float
            )
        )
    )
)


all_outputs_finite = bool(
    np.all(
        np.isfinite(
            analysis[
                OUTPUT_METRICS
            ].to_numpy(
                dtype=float
            )
        )
    )
)


all_derived_finite = bool(
    np.all(
        np.isfinite(
            analysis[
                DERIVED_SENSITIVITY_PREDICTORS
            ].to_numpy(
                dtype=float
            )
        )
    )
)


# ------------------------------------------------------------
# 9. PRIMARY SPEARMAN ANALYSIS
# ------------------------------------------------------------

spearman_table = (
    compute_spearman_screening(
        dataframe=
            analysis,

        input_columns=
            PHASE16_INPUT_PARAMETERS,

        output_columns=
            OUTPUT_METRICS
    )
)


spearman_table[
    "subsystem"
] = (
    spearman_table[
        "parameter"
    ]
    .map(
        PARAMETER_SUBSYSTEM
    )
)


spearman_table.to_csv(
    spearman_path,
    index=False
)


# ------------------------------------------------------------
# 10. PHYSICALLY DERIVED DIAGNOSTICS
# ------------------------------------------------------------

derived_table = (
    compute_spearman_screening(
        dataframe=
            analysis,

        input_columns=
            DERIVED_SENSITIVITY_PREDICTORS,

        output_columns=
            OUTPUT_METRICS
    )
)


derived_table.to_csv(
    derived_path,
    index=False
)


# ------------------------------------------------------------
# 11. MULTIVARIATE RIDGE SCREENING
# ------------------------------------------------------------

(
    ridge_table,
    ridge_model_summary
) = (
    compute_standardized_ridge_screening(
        dataframe=
            analysis,

        input_columns=
            PHASE16_INPUT_PARAMETERS,

        output_columns=
            OUTPUT_METRICS,

        random_state=
            16004
    )
)


ridge_table[
    "subsystem"
] = (
    ridge_table[
        "parameter"
    ]
    .map(
        PARAMETER_SUBSYSTEM
    )
)


ridge_table.to_csv(
    ridge_path,
    index=False
)


ridge_model_summary.to_csv(
    ridge_summary_path,
    index=False
)


# ------------------------------------------------------------
# 12. TOP SPEARMAN FACTORS
# ------------------------------------------------------------

top_factors = (
    build_top_factor_table(
        spearman_table=
            spearman_table,

        top_n=
            5
    )
)


top_factors[
    "subsystem"
] = (
    top_factors[
        "parameter"
    ]
    .map(
        PARAMETER_SUBSYSTEM
    )
)


top_factors.to_csv(
    top_factors_path,
    index=False
)


# ------------------------------------------------------------
# 13. VALIDATE TABLE DIMENSIONS
# ------------------------------------------------------------

expected_spearman_rows = (
    len(
        PHASE16_INPUT_PARAMETERS
    )
    *
    len(
        OUTPUT_METRICS
    )
)


expected_derived_rows = (
    len(
        DERIVED_SENSITIVITY_PREDICTORS
    )
    *
    len(
        OUTPUT_METRICS
    )
)


spearman_dimension_valid = (
    len(
        spearman_table
    )
    ==
    expected_spearman_rows
)


derived_dimension_valid = (
    len(
        derived_table
    )
    ==
    expected_derived_rows
)


ridge_dimension_valid = (
    len(
        ridge_table
    )
    ==
    expected_spearman_rows
)


ridge_models_complete = (
    len(
        ridge_model_summary
    )
    ==
    len(
        OUTPUT_METRICS
    )
)


spearman_finite = bool(
    np.all(
        np.isfinite(
            spearman_table[
                [
                    "spearman_rho",
                    "absolute_rho",
                    "p_value",
                    "q_value"
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)


ridge_finite = bool(
    np.all(
        np.isfinite(
            ridge_table[
                [
                    "standardized_coefficient",
                    "absolute_coefficient"
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)


# ------------------------------------------------------------
# 14. HELPER — GET ONE SPEARMAN RESULT
# ------------------------------------------------------------

def get_spearman(
    table,
    output_metric,
    parameter
):

    row = table[
        (
            table[
                "output_metric"
            ]
            ==
            output_metric
        )
        &
        (
            table[
                "parameter"
            ]
            ==
            parameter
        )
    ]


    if len(
        row
    ) != 1:

        raise RuntimeError(
            f"Cannot find sensitivity pair: "
            f"{output_metric} / {parameter}"
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
                "q_value"
            ]
        )
    )


# ------------------------------------------------------------
# 15. KEY PHYSICAL HYPOTHESES
# ------------------------------------------------------------

(
    rho_fault_noise_fdir,
    q_fault_noise_fdir
) = get_spearman(
    derived_table,
    "fdir_pd",
    "fault_to_noise_ratio"
)


(
    rho_fault_noise_hybrid,
    q_fault_noise_hybrid
) = get_spearman(
    derived_table,
    "hybrid_pd",
    "fault_to_noise_ratio"
)


(
    rho_noise_hybrid_pfa,
    q_noise_hybrid_pfa
) = get_spearman(
    spearman_table,
    "hybrid_pfa",
    "pseudorange_noise_std_m"
)


(
    rho_dual_duration,
    q_dual_duration
) = get_spearman(
    spearman_table,
    "protected_dual_outage_rmse",
    "dual_outage_duration_min"
)


(
    rho_accel_noise_dual,
    q_accel_noise_dual
) = get_spearman(
    spearman_table,
    "protected_dual_outage_rmse",
    "accelerometer_noise_std_mps2"
)


(
    rho_accel_bias_dual,
    q_accel_bias_dual
) = get_spearman(
    derived_table,
    "protected_dual_outage_rmse",
    "accelerometer_bias_norm_mps2"
)


# ------------------------------------------------------------
# 16. STRONG / SIGNIFICANT COUNTS
# ------------------------------------------------------------

strong_sensitivity_count = int(
    np.sum(
        spearman_table[
            "absolute_rho"
        ]
        >=
        0.40
    )
)


fdr_significant_count = int(
    np.sum(
        spearman_table[
            "fdr_significant_0p10"
        ]
    )
)


# ------------------------------------------------------------
# 17. TOP PARAMETER PER OUTPUT
# ------------------------------------------------------------

top_parameter_rows = []


for output_metric in OUTPUT_METRICS:

    subset = (
        spearman_table[
            spearman_table[
                "output_metric"
            ]
            ==
            output_metric
        ]
        .sort_values(
            "absolute_rho",
            ascending=False
        )
    )


    top_row = subset.iloc[
        0
    ]


    ridge_subset = (
        ridge_table[
            ridge_table[
                "output_metric"
            ]
            ==
            output_metric
        ]
        .sort_values(
            "absolute_coefficient",
            ascending=False
        )
    )


    ridge_top_row = (
        ridge_subset.iloc[
            0
        ]
    )


    model_row = (
        ridge_model_summary[
            ridge_model_summary[
                "output_metric"
            ]
            ==
            output_metric
        ]
        .iloc[
            0
        ]
    )


    top_parameter_rows.append(
        {
            "output_metric":
                output_metric,

            "top_spearman_parameter":
                top_row[
                    "parameter"
                ],

            "top_spearman_rho":
                top_row[
                    "spearman_rho"
                ],

            "top_spearman_q":
                top_row[
                    "q_value"
                ],

            "top_ridge_parameter":
                ridge_top_row[
                    "parameter"
                ],

            "top_ridge_coefficient":
                ridge_top_row[
                    "standardized_coefficient"
                ],

            "ridge_cv_r2_mean":
                model_row[
                    "cv_r2_mean"
                ]
        }
    )


top_parameter_summary = pd.DataFrame(
    top_parameter_rows
)


# ------------------------------------------------------------
# 18. SCIENTIFIC FLAGS
#
# These are RESULTS, not validation requirements.
# ------------------------------------------------------------

fault_to_noise_controls_fdir = (
    rho_fault_noise_fdir
    >
    0.50
)


fault_to_noise_controls_hybrid = (
    rho_fault_noise_hybrid
    >
    0.50
)


noise_increases_hybrid_pfa = (
    rho_noise_hybrid_pfa
    >
    0.30
)


dual_duration_degrades_dual_rmse = (
    rho_dual_duration
    >
    0.30
)


# ------------------------------------------------------------
# 19. METHODOLOGICAL VALIDATION
# ------------------------------------------------------------

validation_016d = (
    scenario_sets_identical
    and
    correct_sample_count
    and
    all_inputs_finite
    and
    all_outputs_finite
    and
    all_derived_finite
    and
    spearman_dimension_valid
    and
    derived_dimension_valid
    and
    ridge_dimension_valid
    and
    ridge_models_complete
    and
    spearman_finite
    and
    ridge_finite
)


# ------------------------------------------------------------
# 20. PRINT HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 016-D"
)


print(
    "Multidimensional sensitivity screening"
)


print(
    "========================================================================================================================"
)


print(
    f"Scenarios : "
    f"{len(analysis)}"
)


print(
    f"Independent Phase-16 design variables : "
    f"{len(PHASE16_INPUT_PARAMETERS)}"
)


print(
    f"Output metrics analysed : "
    f"{len(OUTPUT_METRICS)}"
)


print(
    f"Spearman tests : "
    f"{len(spearman_table)}"
)


print(
    f"FDR-significant relationships q <= 0.10 : "
    f"{fdr_significant_count}"
)


print(
    f"Strong |rho| >= 0.40 relationships : "
    f"{strong_sensitivity_count}"
)


print()


# ------------------------------------------------------------
# 21. TOP FACTORS
# ------------------------------------------------------------

print(
    "----- TOP SPEARMAN FACTORS BY OUTPUT -----"
)


for output_metric in OUTPUT_METRICS:

    print()

    print(
        f"{output_metric}"
    )


    subset = (
        top_factors[
            top_factors[
                "output_metric"
            ]
            ==
            output_metric
        ]
        .head(
            5
        )
    )


    print(
        subset[
            [
                "rank",
                "parameter",
                "subsystem",
                "spearman_rho",
                "q_value",
                "fdr_significant_0p10"
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "spearman_rho":
                    lambda value:
                        f"{value:+.3f}",

                "q_value":
                    lambda value:
                        f"{value:.4f}"
            }
        )
    )


# ------------------------------------------------------------
# 22. RIDGE SUMMARY
# ------------------------------------------------------------

print()

print(
    "----- MULTIVARIATE RIDGE DIAGNOSTIC -----"
)


print(
    top_parameter_summary.to_string(
        index=False,
        formatters={
            "top_spearman_rho":
                lambda value:
                    f"{value:+.3f}",

            "top_spearman_q":
                lambda value:
                    f"{value:.4f}",

            "top_ridge_coefficient":
                lambda value:
                    f"{value:+.3f}",

            "ridge_cv_r2_mean":
                lambda value:
                    f"{value:+.3f}"
        }
    )
)


print()


print(
    "NOTE:"
)


print(
    "Negative Ridge CV R^2 is allowed and means that the "
    "23-variable linear-additive model has weak predictive "
    "value for that output under the 64-point campaign."
)


print()


# ------------------------------------------------------------
# 23. PHYSICAL DIAGNOSTICS
# ------------------------------------------------------------

print(
    "----- PHYSICALLY MOTIVATED DIAGNOSTICS -----"
)


print(
    f"fault/noise ratio -> FDIR P_D : "
    f"rho={rho_fault_noise_fdir:+.3f}, "
    f"q={q_fault_noise_fdir:.4f}"
)


print(
    f"fault/noise ratio -> hybrid P_D : "
    f"rho={rho_fault_noise_hybrid:+.3f}, "
    f"q={q_fault_noise_hybrid:.4f}"
)


print(
    f"pseudorange noise -> hybrid P_FA : "
    f"rho={rho_noise_hybrid_pfa:+.3f}, "
    f"q={q_noise_hybrid_pfa:.4f}"
)


print(
    f"dual-outage duration -> protected dual RMSE : "
    f"rho={rho_dual_duration:+.3f}, "
    f"q={q_dual_duration:.4f}"
)


print(
    f"accelerometer noise -> protected dual RMSE : "
    f"rho={rho_accel_noise_dual:+.3f}, "
    f"q={q_accel_noise_dual:.4f}"
)


print(
    f"accelerometer-bias norm -> protected dual RMSE : "
    f"rho={rho_accel_bias_dual:+.3f}, "
    f"q={q_accel_bias_dual:.4f}"
)


print()


# ------------------------------------------------------------
# 24. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Fault/noise ratio strongly associated with FDIR detection : "
    f"{fault_to_noise_controls_fdir}"
)


print(
    f"Fault/noise ratio strongly associated with hybrid detection : "
    f"{fault_to_noise_controls_hybrid}"
)


print(
    f"Higher pseudorange noise associated with higher hybrid P_FA : "
    f"{noise_increases_hybrid_pfa}"
)


print(
    f"Longer dual outage associated with larger protected dual RMSE : "
    f"{dual_duration_degrades_dual_rmse}"
)


print()


# ------------------------------------------------------------
# 25. FILES
# ------------------------------------------------------------

print(
    "----- FILES -----"
)


print(
    f"Analysis dataset : "
    f"{analysis_dataset_path}"
)


print(
    f"Spearman sensitivity : "
    f"{spearman_path}"
)


print(
    f"Derived sensitivity : "
    f"{derived_path}"
)


print(
    f"Ridge sensitivity : "
    f"{ridge_path}"
)


print(
    f"Ridge model summary : "
    f"{ridge_summary_path}"
)


print(
    f"Top factors : "
    f"{top_factors_path}"
)


print()


# ------------------------------------------------------------
# 26. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"016-A/016-B/016-C scenario sets identical : "
    f"{scenario_sets_identical}"
)


print(
    f"Exactly 64 merged scenarios : "
    f"{correct_sample_count}"
)


print(
    f"All input parameters finite : "
    f"{all_inputs_finite}"
)


print(
    f"All output metrics finite : "
    f"{all_outputs_finite}"
)


print(
    f"All derived predictors finite : "
    f"{all_derived_finite}"
)


print(
    f"Spearman table dimension valid : "
    f"{spearman_dimension_valid}"
)


print(
    f"Derived table dimension valid : "
    f"{derived_dimension_valid}"
)


print(
    f"Ridge table dimension valid : "
    f"{ridge_dimension_valid}"
)


print(
    f"All Spearman statistics finite : "
    f"{spearman_finite}"
)


print(
    f"All Ridge coefficients finite : "
    f"{ridge_finite}"
)


print()


print(
    f"VALIDATION GLOBALE 016-D : "
    f"{validation_016d}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 27. FIGURE — SPEARMAN HEATMAP
# ------------------------------------------------------------

spearman_matrix = (
    spearman_table.pivot(
        index=
            "output_metric",

        columns=
            "parameter",

        values=
            "spearman_rho"
    )
    .reindex(
        index=
            OUTPUT_METRICS,

        columns=
            PHASE16_INPUT_PARAMETERS
    )
)


plt.figure(
    figsize=(18, 9)
)


image = plt.imshow(
    spearman_matrix.to_numpy(),
    aspect="auto",
    vmin=-1.0,
    vmax=1.0
)


plt.colorbar(
    image,
    label="Spearman rho"
)


plt.xticks(
    np.arange(
        len(
            PHASE16_INPUT_PARAMETERS
        )
    ),
    PHASE16_INPUT_PARAMETERS,
    rotation=90,
    fontsize=8
)


plt.yticks(
    np.arange(
        len(
            OUTPUT_METRICS
        )
    ),
    OUTPUT_METRICS,
    fontsize=9
)


plt.xlabel(
    "Phase-16 input parameter"
)


plt.ylabel(
    "Output metric"
)


plt.title(
    "AURORA — 016-D Spearman sensitivity screening"
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016d_spearman_heatmap.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 28. FIGURES — TOP FACTORS FOR KEY OUTPUTS
# ------------------------------------------------------------

KEY_FIGURE_METRICS = [

    "protected_mission_rmse",

    "protected_dual_outage_rmse",

    "fdir_pd",

    "hybrid_pd",

    "hybrid_pfa"
]


for output_metric in KEY_FIGURE_METRICS:

    subset = (
        spearman_table[
            spearman_table[
                "output_metric"
            ]
            ==
            output_metric
        ]
        .sort_values(
            "absolute_rho",
            ascending=False
        )
        .head(
            8
        )
        .sort_values(
            "absolute_rho",
            ascending=True
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(
        subset[
            "parameter"
        ],

        subset[
            "spearman_rho"
        ]
    )


    plt.axvline(
        0.0,
        linestyle="--"
    )


    plt.xlabel(
        "Spearman rho"
    )


    plt.ylabel(
        "Parameter"
    )


    plt.title(
        f"AURORA — 016-D sensitivity: {output_metric}"
    )


    plt.grid(
        True,
        axis="x"
    )


    plt.tight_layout()


    plt.savefig(
        figure_directory
        /
        f"phase16_016d_top_{output_metric}.png",

        dpi=200
    )


    plt.show()


# ------------------------------------------------------------
# 29. FIGURE — PHYSICAL FAULT/NOISE RATIO
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    analysis[
        "fault_to_noise_ratio"
    ],

    analysis[
        "fdir_pd"
    ],

    label="Classical FDIR"
)


plt.scatter(
    analysis[
        "fault_to_noise_ratio"
    ],

    analysis[
        "hybrid_pd"
    ],

    label="Hybrid"
)


plt.xlabel(
    "Fault magnitude / pseudorange noise std"
)


plt.ylabel(
    "Epoch detection rate [%]"
)


plt.title(
    "AURORA — 016-D detection vs fault-to-noise ratio"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016d_fault_to_noise_detection.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 30. FIGURE — HYBRID PFA VS GNSS NOISE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    analysis[
        "pseudorange_noise_std_m"
    ],

    analysis[
        "hybrid_pfa"
    ]
)


plt.xlabel(
    "Pseudorange noise std [m]"
)


plt.ylabel(
    "Hybrid false-alarm rate [%]"
)


plt.title(
    "AURORA — 016-D hybrid false alarms vs GNSS noise"
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016d_hybrid_pfa_vs_noise.png",

    dpi=200
)


plt.show()