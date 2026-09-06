from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.validation.robustness_campaign import (
    generate_latin_hypercube_campaign,
    summarize_campaign,
    campaign_values_within_bounds,
    campaign_values_finite,
    latin_hypercube_stratification_valid,
    campaigns_identical,
    phase16_timeline_constraints_valid
)


# ============================================================
# AURORA
# Experience 016-A
#
# LARGE ROBUSTNESS CAMPAIGN DESIGN
#
#
# Goal:
#
# Build the fixed design-of-experiments table used by the
# numerical robustness campaigns of Phase 16.
#
#
# Method:
#
#       Latin Hypercube Sampling
#
# Number of scenarios:
#
#       64
#
#
# Variables:
#
#       GNSS noise
#       accelerometer noise
#       accelerometer bias XYZ
#       gyro noise
#       gyro bias XYZ
#       star tracker noise
#       initial position error XYZ
#       initial velocity error XYZ
#       non-gravitational force magnitude
#       GNSS fault magnitude
#       GNSS fault start time
#       GNSS fault duration
#       GNSS outage duration
#       star-tracker outage duration
#       dual-outage duration
#
#
# Orbit geometry is NOT varied here.
#
# That belongs to Phase 17.
# ============================================================


# ------------------------------------------------------------
# 1. CAMPAIGN SETTINGS
# ------------------------------------------------------------

number_of_scenarios = (
    64
)


campaign_seed = (
    16001
)


# ------------------------------------------------------------
# 2. OUTPUT DIRECTORIES
# ------------------------------------------------------------

data_directory = (
    Path("data")
    /
    "phase16"
)


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


data_directory.mkdir(
    parents=True,
    exist_ok=True
)


table_directory.mkdir(
    parents=True,
    exist_ok=True
)


figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


campaign_path = (
    data_directory
    /
    "robustness_campaign_016a.csv"
)


summary_path = (
    table_directory
    /
    "phase16_016a_parameter_summary.csv"
)


correlation_path = (
    table_directory
    /
    "phase16_016a_parameter_correlations.csv"
)


# ------------------------------------------------------------
# 3. GENERATE CAMPAIGN
# ------------------------------------------------------------

campaign = (
    generate_latin_hypercube_campaign(
        number_of_runs=
            number_of_scenarios,

        seed=
            campaign_seed
    )
)


dataframe = (
    campaign.dataframe
)


parameter_names = [
    spec.name
    for spec in campaign.parameter_specs
]


number_of_parameters = len(
    parameter_names
)


# ------------------------------------------------------------
# 4. REGENERATE WITH SAME SEED
#
# This is an explicit reproducibility test.
# ------------------------------------------------------------

campaign_repeated = (
    generate_latin_hypercube_campaign(
        number_of_runs=
            number_of_scenarios,

        seed=
            campaign_seed
    )
)


reproducibility_valid = (
    campaigns_identical(
        campaign,
        campaign_repeated
    )
)


# ------------------------------------------------------------
# 5. CORE VALIDATION
# ------------------------------------------------------------

number_of_rows_valid = (
    len(
        dataframe
    )
    ==
    number_of_scenarios
)


number_of_parameters_valid = (
    number_of_parameters
    >
    10
)


values_finite = (
    campaign_values_finite(
        campaign
    )
)


values_within_bounds = (
    campaign_values_within_bounds(
        campaign
    )
)


lhs_stratification_valid = (
    latin_hypercube_stratification_valid(
        campaign
    )
)


timeline_valid = (
    phase16_timeline_constraints_valid(
        campaign
    )
)


scenario_ids_unique = (
    dataframe[
        "scenario_id"
    ].nunique()
    ==
    number_of_scenarios
)


scenario_seeds_unique = (
    dataframe[
        "scenario_seed"
    ].nunique()
    ==
    number_of_scenarios
)


# ------------------------------------------------------------
# 6. PARAMETER SUMMARY
# ------------------------------------------------------------

summary = (
    summarize_campaign(
        campaign
    )
)


# ------------------------------------------------------------
# 7. CORRELATION DIAGNOSTIC
#
# LHS guarantees marginal stratification, not zero
# correlation in a finite sample.
#
# Correlation is therefore reported diagnostically,
# not used as a hard pass/fail requirement.
# ------------------------------------------------------------

correlation_matrix = (
    dataframe[
        parameter_names
    ]
    .corr()
)


correlation_rows = []


for first_index in range(
    number_of_parameters
):

    for second_index in range(
        first_index + 1,
        number_of_parameters
    ):

        first_name = (
            parameter_names[
                first_index
            ]
        )


        second_name = (
            parameter_names[
                second_index
            ]
        )


        correlation = float(
            correlation_matrix.loc[
                first_name,
                second_name
            ]
        )


        correlation_rows.append(
            {
                "parameter_1":
                    first_name,

                "parameter_2":
                    second_name,

                "correlation":
                    correlation,

                "absolute_correlation":
                    abs(
                        correlation
                    )
            }
        )


correlation_table = (
    pd.DataFrame(
        correlation_rows
    )
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


maximum_absolute_correlation = float(
    correlation_table[
        "absolute_correlation"
    ].iloc[
        0
    ]
)


# ------------------------------------------------------------
# 8. QUARTILE COVERAGE DIAGNOSTIC
#
# Because this is a Latin Hypercube, each parameter should
# be distributed throughout its allowed interval rather than
# clustering near the center.
# ------------------------------------------------------------

quartile_counts = []


unit_samples = (
    campaign.unit_samples
)


for dimension_index, parameter_name in enumerate(
    parameter_names
):

    values = (
        unit_samples[
            :,
            dimension_index
        ]
    )


    counts = np.histogram(
        values,
        bins=[
            0.0,
            0.25,
            0.50,
            0.75,
            1.0
        ]
    )[0]


    quartile_counts.append(
        {
            "parameter":
                parameter_name,

            "Q1":
                int(
                    counts[
                        0
                    ]
                ),

            "Q2":
                int(
                    counts[
                        1
                    ]
                ),

            "Q3":
                int(
                    counts[
                        2
                    ]
                ),

            "Q4":
                int(
                    counts[
                        3
                    ]
                )
        }
    )


quartile_table = pd.DataFrame(
    quartile_counts
)


expected_per_quartile = (
    number_of_scenarios
    //
    4
)


quartile_coverage_valid = bool(
    np.all(
        quartile_table[
            [
                "Q1",
                "Q2",
                "Q3",
                "Q4"
            ]
        ].to_numpy()
        ==
        expected_per_quartile
    )
)


# ------------------------------------------------------------
# 9. DERIVED PHYSICAL DIAGNOSTICS
# ------------------------------------------------------------

accelerometer_bias_norm = np.sqrt(
    dataframe[
        "accelerometer_bias_x_mps2"
    ].to_numpy()**2
    +
    dataframe[
        "accelerometer_bias_y_mps2"
    ].to_numpy()**2
    +
    dataframe[
        "accelerometer_bias_z_mps2"
    ].to_numpy()**2
)


gyro_bias_norm = np.sqrt(
    dataframe[
        "gyro_bias_x_degps"
    ].to_numpy()**2
    +
    dataframe[
        "gyro_bias_y_degps"
    ].to_numpy()**2
    +
    dataframe[
        "gyro_bias_z_degps"
    ].to_numpy()**2
)


initial_position_error_norm = np.sqrt(
    dataframe[
        "initial_position_error_x_m"
    ].to_numpy()**2
    +
    dataframe[
        "initial_position_error_y_m"
    ].to_numpy()**2
    +
    dataframe[
        "initial_position_error_z_m"
    ].to_numpy()**2
)


initial_velocity_error_norm = np.sqrt(
    dataframe[
        "initial_velocity_error_x_mps"
    ].to_numpy()**2
    +
    dataframe[
        "initial_velocity_error_y_mps"
    ].to_numpy()**2
    +
    dataframe[
        "initial_velocity_error_z_mps"
    ].to_numpy()**2
)


dataframe[
    "accelerometer_bias_norm_mps2"
] = (
    accelerometer_bias_norm
)


dataframe[
    "gyro_bias_norm_degps"
] = (
    gyro_bias_norm
)


dataframe[
    "initial_position_error_norm_m"
] = (
    initial_position_error_norm
)


dataframe[
    "initial_velocity_error_norm_mps"
] = (
    initial_velocity_error_norm
)


# ------------------------------------------------------------
# 10. SAVE FILES
# ------------------------------------------------------------

dataframe.to_csv(
    campaign_path,
    index=False
)


summary.to_csv(
    summary_path,
    index=False
)


correlation_table.to_csv(
    correlation_path,
    index=False
)


# ------------------------------------------------------------
# 11. GLOBAL VALIDATION
# ------------------------------------------------------------

validation_016a = (
    number_of_rows_valid
    and
    number_of_parameters_valid
    and
    values_finite
    and
    values_within_bounds
    and
    lhs_stratification_valid
    and
    quartile_coverage_valid
    and
    timeline_valid
    and
    scenario_ids_unique
    and
    scenario_seeds_unique
    and
    reproducibility_valid
)


# ------------------------------------------------------------
# 12. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 016-A"
)


print(
    "Large robustness campaign design — Latin Hypercube Sampling"
)


print(
    "========================================================================================================================"
)


print(
    f"Campaign seed : "
    f"{campaign_seed}"
)


print(
    f"Scenarios : "
    f"{number_of_scenarios}"
)


print(
    f"Parameters varied : "
    f"{number_of_parameters}"
)


print(
    f"Expected samples per quartile : "
    f"{expected_per_quartile}"
)


print()


# ------------------------------------------------------------
# 13. PARAMETER TABLE
# ------------------------------------------------------------

print(
    "----- PARAMETER SPACE -----"
)


print(
    summary[
        [
            "parameter",
            "unit",
            "scaling",
            "design_lower",
            "sample_min",
            "sample_mean",
            "sample_max",
            "design_upper"
        ]
    ].to_string(
        index=False,
        formatters={
            "design_lower":
                lambda value:
                    f"{value:.6e}",

            "sample_min":
                lambda value:
                    f"{value:.6e}",

            "sample_mean":
                lambda value:
                    f"{value:.6e}",

            "sample_max":
                lambda value:
                    f"{value:.6e}",

            "design_upper":
                lambda value:
                    f"{value:.6e}"
        }
    )
)


print()


# ------------------------------------------------------------
# 14. DERIVED DISTRIBUTIONS
# ------------------------------------------------------------

print(
    "----- DERIVED SCENARIO DIAGNOSTICS -----"
)


print(
    f"Accelerometer bias norm "
    f"min/mean/max : "
    f"{np.min(accelerometer_bias_norm):.3e} / "
    f"{np.mean(accelerometer_bias_norm):.3e} / "
    f"{np.max(accelerometer_bias_norm):.3e} m/s^2"
)


print(
    f"Gyro bias norm "
    f"min/mean/max : "
    f"{np.min(gyro_bias_norm):.6f} / "
    f"{np.mean(gyro_bias_norm):.6f} / "
    f"{np.max(gyro_bias_norm):.6f} deg/s"
)


print(
    f"Initial position error norm "
    f"min/mean/max : "
    f"{np.min(initial_position_error_norm):.2f} / "
    f"{np.mean(initial_position_error_norm):.2f} / "
    f"{np.max(initial_position_error_norm):.2f} m"
)


print(
    f"Initial velocity error norm "
    f"min/mean/max : "
    f"{np.min(initial_velocity_error_norm):.4f} / "
    f"{np.mean(initial_velocity_error_norm):.4f} / "
    f"{np.max(initial_velocity_error_norm):.4f} m/s"
)


print()


# ------------------------------------------------------------
# 15. CORRELATION DIAGNOSTIC
# ------------------------------------------------------------

print(
    "----- LARGEST ABSOLUTE PARAMETER CORRELATIONS -----"
)


print(
    correlation_table[
        [
            "parameter_1",
            "parameter_2",
            "correlation"
        ]
    ]
    .head(
        10
    )
    .to_string(
        index=False,
        formatters={
            "correlation":
                lambda value:
                    f"{value:+.3f}"
        }
    )
)


print()


print(
    f"Maximum absolute pairwise correlation : "
    f"{maximum_absolute_correlation:.3f}"
)


print(
    "NOTE: correlation is diagnostic only; "
    "Latin Hypercube guarantees marginal stratification, "
    "not exact zero correlation."
)


print()


# ------------------------------------------------------------
# 16. OUTPUT FILES
# ------------------------------------------------------------

print(
    "----- FILES -----"
)


print(
    f"Campaign : "
    f"{campaign_path}"
)


print(
    f"Parameter summary : "
    f"{summary_path}"
)


print(
    f"Correlation table : "
    f"{correlation_path}"
)


print()


# ------------------------------------------------------------
# 17. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"Correct number of scenarios : "
    f"{number_of_rows_valid}"
)


print(
    f"Parameter dimension valid : "
    f"{number_of_parameters_valid}"
)


print(
    f"All values finite : "
    f"{values_finite}"
)


print(
    f"All values within design bounds : "
    f"{values_within_bounds}"
)


print(
    f"Latin Hypercube stratification valid : "
    f"{lhs_stratification_valid}"
)


print(
    f"Exactly 16 samples per quartile/dimension : "
    f"{quartile_coverage_valid}"
)


print(
    f"Mission timeline constraints valid : "
    f"{timeline_valid}"
)


print(
    f"Scenario IDs unique : "
    f"{scenario_ids_unique}"
)


print(
    f"Scenario seeds unique : "
    f"{scenario_seeds_unique}"
)


print(
    f"Exact regeneration with same seed : "
    f"{reproducibility_valid}"
)


print()


print(
    f"VALIDATION GLOBALE 016-A : "
    f"{validation_016a}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 18. FIGURE — NORMALIZED PARAMETER COVERAGE
#
# Every horizontal row is a sampled parameter.
# Values are represented in normalized design space [0,1].
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 9)
)


for dimension_index, parameter_name in enumerate(
    parameter_names
):

    plt.scatter(
        unit_samples[
            :,
            dimension_index
        ],

        np.full(
            number_of_scenarios,
            dimension_index
        ),

        s=12
    )


plt.xlabel(
    "Normalized design coordinate"
)


plt.ylabel(
    "Parameter index"
)


plt.title(
    "AURORA — 016-A Latin Hypercube marginal coverage"
)


plt.xlim(
    0.0,
    1.0
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016a_lhs_coverage.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 19. FIGURE — CORRELATION MATRIX
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 10)
)


image = plt.imshow(
    correlation_matrix.to_numpy(),
    aspect="auto",
    vmin=-1.0,
    vmax=1.0
)


plt.colorbar(
    image,
    label="Pearson correlation"
)


plt.xticks(
    np.arange(
        number_of_parameters
    ),
    np.arange(
        number_of_parameters
    ),
    rotation=90
)


plt.yticks(
    np.arange(
        number_of_parameters
    ),
    np.arange(
        number_of_parameters
    )
)


plt.xlabel(
    "Parameter index"
)


plt.ylabel(
    "Parameter index"
)


plt.title(
    "AURORA — 016-A finite-sample parameter correlations"
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase16_016a_parameter_correlation.png",

    dpi=200
)


plt.show()