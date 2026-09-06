from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    t as student_t
)

from src.validation.orbit_resilience_sweep import (
    run_orbit_resilience_scenario
)


# ============================================================
# AURORA
# Experience 017-B
#
# FULL RESILIENCE ACROSS REPRESENTATIVE ORBITAL GEOMETRIES
#
#
# Representative geometries:
#
#   1. Original AURORA reference orbit
#
#   2. Best mean-PDOP geometry from 017-A
#
#   3. Worst PDOP-p95 geometry from 017-A
#
#   4. Median mean-PDOP geometry from 017-A
#
#
# For every geometry:
#
#       16 stochastic replicates
#
# with the SAME seed for a given replicate across all
# geometries.
#
#
# Therefore comparisons between geometries are paired with
# respect to sensor / GNSS noise realization.
#
#
# Total full missions:
#
#       4 * 16 = 64
#
#
# The controlled five-satellite low-redundancy event remains
# part of the mission.
#
# 017-A showed that this condition does NOT occur naturally
# in the current simplified constellation model.
# ============================================================


# ------------------------------------------------------------
# 1. INPUT
# ------------------------------------------------------------

geometry_results_path = (
    Path("results")
    /
    "tables"
    /
    "phase17_017a_orbit_geometry_sweep.csv"
)


if not geometry_results_path.exists():

    raise FileNotFoundError(
        f"017-A results not found: "
        f"{geometry_results_path}"
    )


# ------------------------------------------------------------
# 2. OUTPUT
# ------------------------------------------------------------

data_directory = (
    Path("data")
    /
    "phase17"
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


selected_geometry_path = (
    data_directory
    /
    "representative_orbits_017b.csv"
)


results_path = (
    table_directory
    /
    "phase17_017b_orbit_resilience.csv"
)


summary_path = (
    table_directory
    /
    "phase17_017b_geometry_summary.csv"
)


paired_difference_path = (
    table_directory
    /
    "phase17_017b_paired_differences_vs_reference.csv"
)


# ------------------------------------------------------------
# 3. MONTE CARLO
# ------------------------------------------------------------

NUMBER_OF_REPLICATES = (
    16
)


BASE_SEED = (
    170_200
)


# ------------------------------------------------------------
# 4. LOAD 017-A
# ------------------------------------------------------------

geometry_results = pd.read_csv(
    geometry_results_path
)


# ------------------------------------------------------------
# 5. AUTOMATIC REPRESENTATIVE GEOMETRY SELECTION
# ------------------------------------------------------------

best_index = (
    geometry_results[
        "pdop_mean"
    ].idxmin()
)


worst_index = (
    geometry_results[
        "pdop_p95"
    ].idxmax()
)


median_pdop = float(
    np.median(
        geometry_results[
            "pdop_mean"
        ]
    )
)


median_index = (
    (
        geometry_results[
            "pdop_mean"
        ]
        -
        median_pdop
    )
    .abs()
    .idxmin()
)


best_row = (
    geometry_results.loc[
        best_index
    ]
)


worst_row = (
    geometry_results.loc[
        worst_index
    ]
)


median_row = (
    geometry_results.loc[
        median_index
    ]
)


# ------------------------------------------------------------
# 6. ORIGINAL AURORA REFERENCE
#
# This is the original orbital geometry used throughout
# most of Phases 1-16.
# ------------------------------------------------------------

reference_geometry = {
    "geometry_name":
        "REFERENCE_AURORA",

    "source_scenario_id":
        "ORIGINAL_REFERENCE",

    "altitude_km":
        550.0,

    "inclination_deg":
        97.6,

    "initial_true_anomaly_deg":
        25.0,

    "eccentricity":
        0.01,

    "raan_deg":
        40.0,

    "argument_of_periapsis_deg":
        30.0
}


# ------------------------------------------------------------
# 7. CONVERT 017-A ROW
# ------------------------------------------------------------

def geometry_from_017a_row(
    row,
    geometry_name
):

    return {
        "geometry_name":
            geometry_name,

        "source_scenario_id":
            str(
                row[
                    "scenario_id"
                ]
            ),

        "altitude_km":
            float(
                row[
                    "altitude_km"
                ]
            ),

        "inclination_deg":
            float(
                row[
                    "inclination_deg"
                ]
            ),

        "initial_true_anomaly_deg":
            float(
                row[
                    "initial_true_anomaly_deg"
                ]
            ),

        "eccentricity":
            0.01,

        "raan_deg":
            40.0,

        "argument_of_periapsis_deg":
            30.0,

        "source_pdop_mean":
            float(
                row[
                    "pdop_mean"
                ]
            ),

        "source_pdop_p95":
            float(
                row[
                    "pdop_p95"
                ]
            )
    }


selected_geometries = [
    reference_geometry,

    geometry_from_017a_row(
        best_row,
        "BEST_MEAN_PDOP"
    ),

    geometry_from_017a_row(
        worst_row,
        "WORST_PDOP_P95"
    ),

    geometry_from_017a_row(
        median_row,
        "MEDIAN_PDOP"
    )
]


selected_geometry_dataframe = pd.DataFrame(
    selected_geometries
)


selected_geometry_dataframe.to_csv(
    selected_geometry_path,
    index=False
)


# ------------------------------------------------------------
# 8. CHECK UNIQUE GEOMETRIES
# ------------------------------------------------------------

geometry_names = (
    selected_geometry_dataframe[
        "geometry_name"
    ].tolist()
)


geometry_names_unique = (
    len(
        set(
            geometry_names
        )
    )
    ==
    len(
        geometry_names
    )
)


# ------------------------------------------------------------
# 9. CHECKPOINT
# ------------------------------------------------------------

if results_path.exists():

    existing_results = pd.read_csv(
        results_path
    )


    completed_keys = set(
        zip(
            existing_results[
                "geometry_name"
            ].astype(
                str
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


# ------------------------------------------------------------
# 10. BUILD RUN PLAN
# ------------------------------------------------------------

run_plan = []


for replicate_index in range(
    NUMBER_OF_REPLICATES
):

    scenario_seed = (
        BASE_SEED
        +
        replicate_index
    )


    for geometry in selected_geometries:

        key = (
            geometry[
                "geometry_name"
            ],
            replicate_index
        )


        if key in completed_keys:

            continue


        run_plan.append(
            (
                geometry,
                replicate_index,
                scenario_seed
            )
        )


total_expected_runs = (
    len(
        selected_geometries
    )
    *
    NUMBER_OF_REPLICATES
)


# ------------------------------------------------------------
# 11. HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-B"
)


print(
    "Full resilience across representative orbital geometries"
)


print(
    "========================================================================================================================"
)


print(
    f"Geometries : "
    f"{len(selected_geometries)}"
)


print(
    f"Replicates per geometry : "
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
    f"{len(run_plan)}"
)


print()


print(
    "----- SELECTED GEOMETRIES -----"
)


print(
    selected_geometry_dataframe[
        [
            "geometry_name",
            "source_scenario_id",
            "altitude_km",
            "inclination_deg",
            "initial_true_anomaly_deg"
        ]
    ].to_string(
        index=False
    )
)


print()


# ------------------------------------------------------------
# 12. RUN
# ------------------------------------------------------------

new_rows = []


for local_index, (
    geometry,
    replicate_index,
    scenario_seed
) in enumerate(
    run_plan
):

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
        f"{geometry['geometry_name']} | "
        f"rep={replicate_index:02d} | "
        f"h={geometry['altitude_km']:.0f} km | "
        f"i={geometry['inclination_deg']:.1f} deg | "
        f"nu0={geometry['initial_true_anomaly_deg']:.1f} deg | "
        f"seed={scenario_seed}"
    )


    result = (
        run_orbit_resilience_scenario(
            orbit_scenario=
                geometry,

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
                "geometry_name"
            ]
        )
        .drop_duplicates(
            subset=[
                "geometry_name",
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
        f"    P_D low="
        f"{result['low_detection_rate']:.1f}% | "
        f"Prot mission="
        f"{result['protected_mission_rmse']:.3f} m | "
        f"Prot low="
        f"{result['protected_low_redundancy_rmse']:.3f} m | "
        f"Prot dual="
        f"{result['protected_dual_outage_rmse']:.3f} m | "
        f"NIS="
        f"{result['protected_nis_mean']:.3f}"
    )


# ------------------------------------------------------------
# 13. RELOAD
# ------------------------------------------------------------

results = pd.read_csv(
    results_path
)


results = (
    results.sort_values(
        [
            "replicate_index",
            "geometry_name"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# 14. VALIDATION
# ------------------------------------------------------------

all_runs_complete = (
    len(
        results
    )
    ==
    total_expected_runs
)


unique_run_keys = (
    results[
        [
            "geometry_name",
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


all_geometries_complete = bool(
    np.all(
        results.groupby(
            "geometry_name"
        )[
            "replicate_index"
        ]
        .nunique()
        ==
        NUMBER_OF_REPLICATES
    )
)


time_sync_valid = bool(
    np.max(
        np.abs(
            results[
                "time_sync_error_s"
            ]
        )
    )
    <
    1.0e-12
)


core_columns = [
    "strong_detection_rate",
    "strong_isolation_rate",
    "low_detection_rate",
    "false_alarm_rate",

    "attitude_rmse_deg",

    "permissive_mission_rmse",
    "protected_mission_rmse",

    "permissive_low_redundancy_rmse",
    "protected_low_redundancy_rmse",

    "permissive_dual_outage_rmse",
    "protected_dual_outage_rmse",

    "permissive_nis_mean",
    "protected_nis_mean"
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
# 15. SUMMARY
# ------------------------------------------------------------

summary = (
    results.groupby(
        "geometry_name",
        as_index=False
    )
    .agg(
        runs=(
            "replicate_index",
            "size"
        ),

        altitude_km=(
            "altitude_km",
            "first"
        ),

        inclination_deg=(
            "inclination_deg",
            "first"
        ),

        initial_true_anomaly_deg=(
            "initial_true_anomaly_deg",
            "first"
        ),

        natural_visible_mean=(
            "natural_visible_satellites_mean",
            "mean"
        ),

        strong_detection_mean=(
            "strong_detection_rate",
            "mean"
        ),

        low_detection_mean=(
            "low_detection_rate",
            "mean"
        ),

        false_alarm_mean=(
            "false_alarm_rate",
            "mean"
        ),

        attitude_rmse_mean_deg=(
            "attitude_rmse_deg",
            "mean"
        ),

        permissive_mission_rmse_mean=(
            "permissive_mission_rmse",
            "mean"
        ),

        protected_mission_rmse_mean=(
            "protected_mission_rmse",
            "mean"
        ),

        protected_mission_rmse_std=(
            "protected_mission_rmse",
            "std"
        ),

        protected_low_rmse_mean=(
            "protected_low_redundancy_rmse",
            "mean"
        ),

        protected_low_rmse_std=(
            "protected_low_redundancy_rmse",
            "std"
        ),

        protected_dual_rmse_mean=(
            "protected_dual_outage_rmse",
            "mean"
        ),

        protected_dual_rmse_std=(
            "protected_dual_outage_rmse",
            "std"
        ),

        permissive_nis_mean=(
            "permissive_nis_mean",
            "mean"
        ),

        protected_nis_mean=(
            "protected_nis_mean",
            "mean"
        ),

        gain_mission_mean=(
            "gain_mission_rmse_m",
            "mean"
        ),

        gain_low_mean=(
            "gain_low_rmse_m",
            "mean"
        ),

        gain_dual_mean=(
            "gain_dual_rmse_m",
            "mean"
        )
    )
)


summary.to_csv(
    summary_path,
    index=False
)


# ------------------------------------------------------------
# 16. PAIRED DIFFERENCES VS REFERENCE
# ------------------------------------------------------------

reference_results = results[
    results[
        "geometry_name"
    ]
    ==
    "REFERENCE_AURORA"
][
    [
        "replicate_index",
        "protected_mission_rmse",
        "protected_low_redundancy_rmse",
        "protected_dual_outage_rmse",
        "protected_nis_mean"
    ]
].copy()


reference_results = (
    reference_results.rename(
        columns={
            "protected_mission_rmse":
                "reference_mission_rmse",

            "protected_low_redundancy_rmse":
                "reference_low_rmse",

            "protected_dual_outage_rmse":
                "reference_dual_rmse",

            "protected_nis_mean":
                "reference_nis"
        }
    )
)


comparison_results = results[
    results[
        "geometry_name"
    ]
    !=
    "REFERENCE_AURORA"
].copy()


paired = comparison_results.merge(
    reference_results,
    on=
        "replicate_index",
    how=
        "left",
    validate=
        "many_to_one"
)


paired[
    "delta_mission_rmse_m"
] = (
    paired[
        "protected_mission_rmse"
    ]
    -
    paired[
        "reference_mission_rmse"
    ]
)


paired[
    "delta_low_rmse_m"
] = (
    paired[
        "protected_low_redundancy_rmse"
    ]
    -
    paired[
        "reference_low_rmse"
    ]
)


paired[
    "delta_dual_rmse_m"
] = (
    paired[
        "protected_dual_outage_rmse"
    ]
    -
    paired[
        "reference_dual_rmse"
    ]
)


paired[
    "delta_nis"
] = (
    paired[
        "protected_nis_mean"
    ]
    -
    paired[
        "reference_nis"
    ]
)


paired.to_csv(
    paired_difference_path,
    index=False
)


# ------------------------------------------------------------
# 17. CI HELPER
# ------------------------------------------------------------

def mean_ci95(
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


    mean_value = float(
        np.mean(
            values
        )
    )


    if len(
        values
    ) < 2:

        return (
            mean_value,
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
            len(
                values
            )
        )
    )


    critical_value = (
        student_t.ppf(
            0.975,
            df=
                len(
                    values
                )
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
        lower,
        upper
    )


# ------------------------------------------------------------
# 18. PAIRED SUMMARY
# ------------------------------------------------------------

paired_summary_rows = []


for geometry_name in [
    name
    for name in geometry_names
    if name
    !=
    "REFERENCE_AURORA"
]:

    subset = paired[
        paired[
            "geometry_name"
        ]
        ==
        geometry_name
    ]


    row = {
        "geometry_name":
            geometry_name
    }


    for metric_name in [
        "delta_mission_rmse_m",
        "delta_low_rmse_m",
        "delta_dual_rmse_m",
        "delta_nis"
    ]:

        (
            mean_value,
            lower,
            upper
        ) = mean_ci95(
            subset[
                metric_name
            ]
        )


        row[
            f"{metric_name}_mean"
        ] = (
            mean_value
        )


        row[
            f"{metric_name}_ci95_lower"
        ] = (
            lower
        )


        row[
            f"{metric_name}_ci95_upper"
        ] = (
            upper
        )


    paired_summary_rows.append(
        row
    )


paired_summary = pd.DataFrame(
    paired_summary_rows
)


# ------------------------------------------------------------
# 19. SCIENTIFIC FLAGS
# ------------------------------------------------------------

geometry_changes_mission_rmse = bool(
    np.any(
        (
            paired_summary[
                "delta_mission_rmse_m_ci95_lower"
            ]
            >
            0.0
        )
        |
        (
            paired_summary[
                "delta_mission_rmse_m_ci95_upper"
            ]
            <
            0.0
        )
    )
)


geometry_changes_dual_rmse = bool(
    np.any(
        (
            paired_summary[
                "delta_dual_rmse_m_ci95_lower"
            ]
            >
            0.0
        )
        |
        (
            paired_summary[
                "delta_dual_rmse_m_ci95_upper"
            ]
            <
            0.0
        )
    )
)


protected_nis_reasonable = bool(
    np.all(
        (
            summary[
                "protected_nis_mean"
            ]
            >=
            1.5
        )
        &
        (
            summary[
                "protected_nis_mean"
            ]
            <=
            4.5
        )
    )
)


# ------------------------------------------------------------
# 20. VALIDATION
# ------------------------------------------------------------

validation_017b = (
    geometry_names_unique
    and
    all_runs_complete
    and
    unique_run_keys
    and
    all_geometries_complete
    and
    time_sync_valid
    and
    all_core_metrics_finite
)


# ------------------------------------------------------------
# 21. PRINT RESULTS
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-B RESULTS"
)


print(
    "Full resilience across representative orbital geometries"
)


print(
    "========================================================================================================================"
)


print(
    f"Full missions completed : "
    f"{len(results)}/{total_expected_runs}"
)


print(
    f"Replicates per geometry : "
    f"{NUMBER_OF_REPLICATES}"
)


print()


print(
    "----- GEOMETRY SUMMARY -----"
)


print(
    summary.to_string(
        index=False,
        formatters={
            column:
                lambda value:
                    f"{value:.3f}"

            for column in summary.columns

            if column
            not in [
                "geometry_name",
                "runs"
            ]
        }
    )
)


print()


print(
    "----- PAIRED DIFFERENCES VS ORIGINAL AURORA REFERENCE -----"
)


print(
    paired_summary.to_string(
        index=False,
        formatters={
            column:
                lambda value:
                    f"{value:+.4f}"

            for column in paired_summary.columns

            if column
            !=
            "geometry_name"
        }
    )
)


print()


# ------------------------------------------------------------
# 22. BEST / WORST PROTECTED PERFORMANCE
# ------------------------------------------------------------

best_mission_index = (
    summary[
        "protected_mission_rmse_mean"
    ].idxmin()
)


worst_mission_index = (
    summary[
        "protected_mission_rmse_mean"
    ].idxmax()
)


best_dual_index = (
    summary[
        "protected_dual_rmse_mean"
    ].idxmin()
)


worst_dual_index = (
    summary[
        "protected_dual_rmse_mean"
    ].idxmax()
)


print(
    "----- BEST / WORST FULL-MISSION PERFORMANCE -----"
)


print(
    f"Best protected mission RMSE : "
    f"{summary.loc[best_mission_index, 'geometry_name']} | "
    f"{summary.loc[best_mission_index, 'protected_mission_rmse_mean']:.3f} m"
)


print(
    f"Worst protected mission RMSE : "
    f"{summary.loc[worst_mission_index, 'geometry_name']} | "
    f"{summary.loc[worst_mission_index, 'protected_mission_rmse_mean']:.3f} m"
)


print(
    f"Best protected dual-outage RMSE : "
    f"{summary.loc[best_dual_index, 'geometry_name']} | "
    f"{summary.loc[best_dual_index, 'protected_dual_rmse_mean']:.3f} m"
)


print(
    f"Worst protected dual-outage RMSE : "
    f"{summary.loc[worst_dual_index, 'geometry_name']} | "
    f"{summary.loc[worst_dual_index, 'protected_dual_rmse_mean']:.3f} m"
)


print()


# ------------------------------------------------------------
# 23. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f"Orbital geometry measurably changes protected mission RMSE : "
    f"{geometry_changes_mission_rmse}"
)


print(
    f"Orbital geometry measurably changes protected dual-outage RMSE : "
    f"{geometry_changes_dual_rmse}"
)


print(
    f"Protected mean NIS remains in [1.5, 4.5] for all geometries : "
    f"{protected_nis_reasonable}"
)


print()


# ------------------------------------------------------------
# 24. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"Selected geometry names unique : "
    f"{geometry_names_unique}"
)


print(
    f"All 64 full missions complete : "
    f"{all_runs_complete}"
)


print(
    f"All geometry/replicate keys unique : "
    f"{unique_run_keys}"
)


print(
    f"16 replicates available for every geometry : "
    f"{all_geometries_complete}"
)


print(
    f"Time synchronization valid : "
    f"{time_sync_valid}"
)


print(
    f"All core metrics finite : "
    f"{all_core_metrics_finite}"
)


print()


print(
    f"VALIDATION GLOBALE 017-B : "
    f"{validation_017b}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 25. FIGURE — MISSION RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.errorbar(
    summary[
        "geometry_name"
    ],

    summary[
        "protected_mission_rmse_mean"
    ],

    yerr=
        summary[
            "protected_mission_rmse_std"
        ],

    marker="o",
    linestyle="none"
)


plt.ylabel(
    "Protected mission RMSE [m]"
)


plt.xlabel(
    "Orbital geometry"
)


plt.title(
    "AURORA — 017-B protected mission robustness"
)


plt.grid(
    True
)


plt.xticks(
    rotation=20
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017b_mission_rmse_by_geometry.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 26. FIGURE — DUAL OUTAGE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.errorbar(
    summary[
        "geometry_name"
    ],

    summary[
        "protected_dual_rmse_mean"
    ],

    yerr=
        summary[
            "protected_dual_rmse_std"
        ],

    marker="o",
    linestyle="none"
)


plt.ylabel(
    "Protected dual-outage RMSE [m]"
)


plt.xlabel(
    "Orbital geometry"
)


plt.title(
    "AURORA — 017-B dual-outage resilience vs orbit geometry"
)


plt.grid(
    True
)


plt.xticks(
    rotation=20
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017b_dual_rmse_by_geometry.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 27. FIGURE — NIS
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


plt.plot(
    summary[
        "geometry_name"
    ],

    summary[
        "protected_nis_mean"
    ],

    marker="o"
)


plt.axhline(
    3.0,
    linestyle="--",
    label="Expected NIS = 3"
)


plt.ylabel(
    "Mean protected navigation NIS"
)


plt.xlabel(
    "Orbital geometry"
)


plt.title(
    "AURORA — 017-B estimator consistency vs orbital geometry"
)


plt.grid(
    True
)


plt.legend()


plt.xticks(
    rotation=20
)


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017b_nis_by_geometry.png",

    dpi=200
)


plt.show()