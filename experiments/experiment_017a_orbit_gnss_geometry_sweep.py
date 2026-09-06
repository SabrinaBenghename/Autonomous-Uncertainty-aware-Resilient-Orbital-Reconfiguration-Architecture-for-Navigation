from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.validation.orbit_geometry_sweep import (
    ALTITUDES_KM,
    INCLINATIONS_DEG,
    INITIAL_PHASES_DEG,
    generate_phase17_geometry_scenarios,
    run_orbit_geometry_scenario
)


# ============================================================
# AURORA
# Experience 017-A
#
# ORBITAL GEOMETRY + GNSS VISIBILITY SWEEP
#
#
# Sweep:
#
#       altitude:
#           400 / 550 / 700 km
#
#       inclination:
#           0 / 30 / 60 / 75 / 90 / 97.6 deg
#
#       initial true anomaly:
#           0 / 60 / 120 / 180 / 240 / 300 deg
#
#
# Total:
#
#       3 * 6 * 6
#       =
#       108 orbital scenarios
#
#
# Every scenario:
#
#       2 orbital periods
#       ~30 s geometry sampling
#
#
# Metrics:
#
#       visible satellites
#       availability >= 4
#       availability >= 6
#       PDOP distribution
#       longest redundancy gaps
#
#
# >= 6 is the AURORA current FDIR-integrity
# redundancy requirement, NOT a universal GNSS standard.
# ============================================================


# ------------------------------------------------------------
# 1. OUTPUT
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


scenario_design_path = (
    data_directory
    /
    "orbit_geometry_scenarios_017a.csv"
)


results_path = (
    table_directory
    /
    "phase17_017a_orbit_geometry_sweep.csv"
)


altitude_summary_path = (
    table_directory
    /
    "phase17_017a_altitude_summary.csv"
)


inclination_summary_path = (
    table_directory
    /
    "phase17_017a_inclination_summary.csv"
)


worst_geometry_path = (
    table_directory
    /
    "phase17_017a_worst_geometries.csv"
)


# ------------------------------------------------------------
# 2. GENERATE FIXED DESIGN
# ------------------------------------------------------------

scenarios = (
    generate_phase17_geometry_scenarios()
)


scenarios.to_csv(
    scenario_design_path,
    index=False
)


expected_scenario_count = (
    len(
        ALTITUDES_KM
    )
    *
    len(
        INCLINATIONS_DEG
    )
    *
    len(
        INITIAL_PHASES_DEG
    )
)


design_count_valid = (
    len(
        scenarios
    )
    ==
    expected_scenario_count
)


design_ids_unique = (
    scenarios[
        "scenario_id"
    ].nunique()
    ==
    expected_scenario_count
)


# ------------------------------------------------------------
# 3. CHECKPOINT SUPPORT
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


remaining_scenarios = scenarios[
    ~scenarios[
        "scenario_id"
    ].astype(
        str
    ).isin(
        completed_ids
    )
].copy()


# ------------------------------------------------------------
# 4. HEADER
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-A"
)


print(
    "Orbital geometry + GNSS visibility / PDOP sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Altitudes : "
    f"{ALTITUDES_KM} km"
)


print(
    f"Inclinations : "
    f"{INCLINATIONS_DEG} deg"
)


print(
    f"Initial phases : "
    f"{INITIAL_PHASES_DEG} deg"
)


print(
    f"Total scenarios : "
    f"{expected_scenario_count}"
)


print(
    f"Already completed : "
    f"{len(completed_ids)}"
)


print(
    f"Remaining : "
    f"{len(remaining_scenarios)}"
)


print()


# ------------------------------------------------------------
# 5. RUN
# ------------------------------------------------------------

new_rows = []


for local_index, (_, scenario) in enumerate(
    remaining_scenarios.iterrows()
):

    global_index = (
        len(
            completed_ids
        )
        +
        local_index
        +
        1
    )


    print(
        f"[{global_index:03d}/{expected_scenario_count}] "
        f"{scenario['scenario_id']} | "
        f"h={scenario['altitude_km']:.0f} km | "
        f"i={scenario['inclination_deg']:.1f} deg | "
        f"nu0={scenario['initial_true_anomaly_deg']:.0f} deg"
    )


    result = (
        run_orbit_geometry_scenario(
            scenario=
                scenario,

            number_of_orbits=
                2.0,

            geometry_step_seconds=
                30.0,

            pseudorange_noise_std_m=
                3.0
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
        f"    Nsat min/mean/max = "
        f"{result['visible_satellites_min']} / "
        f"{result['visible_satellites_mean']:.2f} / "
        f"{result['visible_satellites_max']} | "
        f"A>=6 = "
        f"{result['availability_ge6_percent']:.2f}% | "
        f"PDOP mean/p95 = "
        f"{result['pdop_mean']:.3f} / "
        f"{result['pdop_p95']:.3f}"
    )


# ------------------------------------------------------------
# 6. RELOAD COMPLETE RESULTS
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
# 7. VALIDATION
# ------------------------------------------------------------

all_scenarios_complete = (
    len(
        results
    )
    ==
    expected_scenario_count
    and
    results[
        "scenario_id"
    ].nunique()
    ==
    expected_scenario_count
)


indices_complete = bool(
    np.array_equal(
        np.sort(
            results[
                "scenario_index"
            ].to_numpy(
                dtype=int
            )
        ),
        np.arange(
            expected_scenario_count,
            dtype=int
        )
    )
)


core_metric_columns = [
    "orbital_period_min",

    "visible_satellites_min",
    "visible_satellites_mean",
    "visible_satellites_max",

    "availability_ge4_percent",
    "availability_ge6_percent",

    "pdop_availability_percent",

    "pdop_min",
    "pdop_mean",
    "pdop_p90",
    "pdop_p95",
    "pdop_max",

    "longest_below4_min",
    "longest_below6_min"
]


all_core_metrics_finite = bool(
    np.all(
        np.isfinite(
            results[
                core_metric_columns
            ].to_numpy(
                dtype=float
            )
        )
    )
)


no_geometry_failures = bool(
    np.all(
        results[
            "geometry_failures"
        ].to_numpy(
            dtype=int
        )
        ==
        0
    )
)


availability_bounds_valid = bool(
    np.all(
        (
            results[
                "availability_ge4_percent"
            ]
            >=
            0.0
        )
        &
        (
            results[
                "availability_ge4_percent"
            ]
            <=
            100.0
        )
    )
    and
    np.all(
        (
            results[
                "availability_ge6_percent"
            ]
            >=
            0.0
        )
        &
        (
            results[
                "availability_ge6_percent"
            ]
            <=
            100.0
        )
    )
)


satellite_count_order_valid = bool(
    np.all(
        results[
            "visible_satellites_min"
        ]
        <=
        results[
            "visible_satellites_mean"
        ]
    )
    and
    np.all(
        results[
            "visible_satellites_mean"
        ]
        <=
        results[
            "visible_satellites_max"
        ]
    )
)


pdop_order_valid = bool(
    np.all(
        results[
            "pdop_min"
        ]
        <=
        results[
            "pdop_mean"
        ]
    )
    and
    np.all(
        results[
            "pdop_mean"
        ]
        <=
        results[
            "pdop_max"
        ]
    )
)


# ------------------------------------------------------------
# 8. GROUPED SUMMARIES
# ------------------------------------------------------------

altitude_summary = (
    results.groupby(
        "altitude_km",
        as_index=False
    )
    .agg(
        scenarios=(
            "scenario_id",
            "size"
        ),

        visible_mean=(
            "visible_satellites_mean",
            "mean"
        ),

        visible_min_worst=(
            "visible_satellites_min",
            "min"
        ),

        availability_ge4_mean=(
            "availability_ge4_percent",
            "mean"
        ),

        availability_ge6_mean=(
            "availability_ge6_percent",
            "mean"
        ),

        availability_ge6_min=(
            "availability_ge6_percent",
            "min"
        ),

        pdop_mean=(
            "pdop_mean",
            "mean"
        ),

        pdop_p95_mean=(
            "pdop_p95",
            "mean"
        ),

        pdop_p95_worst=(
            "pdop_p95",
            "max"
        ),

        longest_below6_worst_min=(
            "longest_below6_min",
            "max"
        )
    )
)


inclination_summary = (
    results.groupby(
        "inclination_deg",
        as_index=False
    )
    .agg(
        scenarios=(
            "scenario_id",
            "size"
        ),

        visible_mean=(
            "visible_satellites_mean",
            "mean"
        ),

        visible_min_worst=(
            "visible_satellites_min",
            "min"
        ),

        availability_ge4_mean=(
            "availability_ge4_percent",
            "mean"
        ),

        availability_ge6_mean=(
            "availability_ge6_percent",
            "mean"
        ),

        availability_ge6_min=(
            "availability_ge6_percent",
            "min"
        ),

        pdop_mean=(
            "pdop_mean",
            "mean"
        ),

        pdop_p95_mean=(
            "pdop_p95",
            "mean"
        ),

        pdop_p95_worst=(
            "pdop_p95",
            "max"
        ),

        longest_below6_worst_min=(
            "longest_below6_min",
            "max"
        )
    )
)


altitude_summary.to_csv(
    altitude_summary_path,
    index=False
)


inclination_summary.to_csv(
    inclination_summary_path,
    index=False
)


# ------------------------------------------------------------
# 9. WORST GEOMETRIES
#
# Ranking priority:
#
#   1. low >=6 availability
#   2. high PDOP p95
# ------------------------------------------------------------

worst_geometries = (
    results.sort_values(
        by=[
            "availability_ge6_percent",
            "pdop_p95"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(
        12
    )
    .copy()
)


worst_geometries.to_csv(
    worst_geometry_path,
    index=False
)


# ------------------------------------------------------------
# 10. BEST / WORST SCENARIOS
# ------------------------------------------------------------

best_integrity_index = (
    results[
        "availability_ge6_percent"
    ].idxmax()
)


worst_integrity_index = (
    results[
        "availability_ge6_percent"
    ].idxmin()
)


best_pdop_index = (
    results[
        "pdop_mean"
    ].idxmin()
)


worst_pdop_index = (
    results[
        "pdop_p95"
    ].idxmax()
)


best_integrity = results.loc[
    best_integrity_index
]


worst_integrity = results.loc[
    worst_integrity_index
]


best_pdop = results.loc[
    best_pdop_index
]


worst_pdop = results.loc[
    worst_pdop_index
]


# ------------------------------------------------------------
# 11. GLOBAL DISTRIBUTION
# ------------------------------------------------------------

global_visible_mean = float(
    np.mean(
        results[
            "visible_satellites_mean"
        ]
    )
)


global_ge4_mean = float(
    np.mean(
        results[
            "availability_ge4_percent"
        ]
    )
)


global_ge6_mean = float(
    np.mean(
        results[
            "availability_ge6_percent"
        ]
    )
)


global_pdop_mean = float(
    np.mean(
        results[
            "pdop_mean"
        ]
    )
)


global_pdop_p95_mean = float(
    np.mean(
        results[
            "pdop_p95"
        ]
    )
)


# ------------------------------------------------------------
# 12. SCIENTIFIC FLAGS
#
# Findings only, not validation conditions.
# ------------------------------------------------------------

all_ge4_full_availability = bool(
    np.all(
        np.isclose(
            results[
                "availability_ge4_percent"
            ],
            100.0
        )
    )
)


all_ge6_full_availability = bool(
    np.all(
        np.isclose(
            results[
                "availability_ge6_percent"
            ],
            100.0
        )
    )
)


phase_matters_for_ge6 = (
    results.groupby(
        [
            "altitude_km",
            "inclination_deg"
        ]
    )[
        "availability_ge6_percent"
    ]
    .std()
    .fillna(
        0.0
    )
    .max()
    >
    0.1
)


# ------------------------------------------------------------
# 13. GLOBAL VALIDATION
# ------------------------------------------------------------

validation_017a = (
    design_count_valid
    and
    design_ids_unique
    and
    all_scenarios_complete
    and
    indices_complete
    and
    all_core_metrics_finite
    and
    no_geometry_failures
    and
    availability_bounds_valid
    and
    satellite_count_order_valid
    and
    pdop_order_valid
)


# ------------------------------------------------------------
# 14. PRINT
# ------------------------------------------------------------

print()

print(
    "========================================================================================================================"
)


print(
    "AURORA — Experience 017-A RESULTS"
)


print(
    "Orbital geometry + GNSS visibility / PDOP sweep"
)


print(
    "========================================================================================================================"
)


print(
    f"Scenarios completed : "
    f"{len(results)}/{expected_scenario_count}"
)


print(
    f"Altitudes : "
    f"{len(ALTITUDES_KM)}"
)


print(
    f"Inclinations : "
    f"{len(INCLINATIONS_DEG)}"
)


print(
    f"Initial phases : "
    f"{len(INITIAL_PHASES_DEG)}"
)


print()


# ------------------------------------------------------------
# 15. GLOBAL GEOMETRY
# ------------------------------------------------------------

print(
    "----- GLOBAL GEOMETRY -----"
)


print(
    f"Mean visible satellites across scenarios : "
    f"{global_visible_mean:.2f}"
)


print(
    f"Mean >=4 availability : "
    f"{global_ge4_mean:.2f} %"
)


print(
    f"Mean >=6 AURORA redundancy availability : "
    f"{global_ge6_mean:.2f} %"
)


print(
    f"Mean PDOP across scenarios : "
    f"{global_pdop_mean:.3f}"
)


print(
    f"Mean scenario PDOP p95 : "
    f"{global_pdop_p95_mean:.3f}"
)


print()


# ------------------------------------------------------------
# 16. ALTITUDE SUMMARY
# ------------------------------------------------------------

print(
    "----- BY ALTITUDE -----"
)


print(
    altitude_summary.to_string(
        index=False,
        formatters={
            "altitude_km":
                lambda value:
                    f"{value:.0f}",

            "visible_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge4_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge6_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge6_min":
                lambda value:
                    f"{value:.2f}",

            "pdop_mean":
                lambda value:
                    f"{value:.3f}",

            "pdop_p95_mean":
                lambda value:
                    f"{value:.3f}",

            "pdop_p95_worst":
                lambda value:
                    f"{value:.3f}",

            "longest_below6_worst_min":
                lambda value:
                    f"{value:.2f}"
        }
    )
)


print()


# ------------------------------------------------------------
# 17. INCLINATION SUMMARY
# ------------------------------------------------------------

print(
    "----- BY INCLINATION -----"
)


print(
    inclination_summary.to_string(
        index=False,
        formatters={
            "inclination_deg":
                lambda value:
                    f"{value:.1f}",

            "visible_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge4_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge6_mean":
                lambda value:
                    f"{value:.2f}",

            "availability_ge6_min":
                lambda value:
                    f"{value:.2f}",

            "pdop_mean":
                lambda value:
                    f"{value:.3f}",

            "pdop_p95_mean":
                lambda value:
                    f"{value:.3f}",

            "pdop_p95_worst":
                lambda value:
                    f"{value:.3f}",

            "longest_below6_worst_min":
                lambda value:
                    f"{value:.2f}"
        }
    )
)


print()


# ------------------------------------------------------------
# 18. BEST / WORST
# ------------------------------------------------------------

print(
    "----- BEST / WORST GEOMETRIES -----"
)


print(
    f"Best >=6 availability : "
    f"{best_integrity['scenario_id']} | "
    f"h={best_integrity['altitude_km']:.0f} km | "
    f"i={best_integrity['inclination_deg']:.1f} deg | "
    f"nu0={best_integrity['initial_true_anomaly_deg']:.0f} deg | "
    f"A6={best_integrity['availability_ge6_percent']:.2f}%"
)


print(
    f"Worst >=6 availability : "
    f"{worst_integrity['scenario_id']} | "
    f"h={worst_integrity['altitude_km']:.0f} km | "
    f"i={worst_integrity['inclination_deg']:.1f} deg | "
    f"nu0={worst_integrity['initial_true_anomaly_deg']:.0f} deg | "
    f"A6={worst_integrity['availability_ge6_percent']:.2f}%"
)


print(
    f"Best mean PDOP : "
    f"{best_pdop['scenario_id']} | "
    f"h={best_pdop['altitude_km']:.0f} km | "
    f"i={best_pdop['inclination_deg']:.1f} deg | "
    f"nu0={best_pdop['initial_true_anomaly_deg']:.0f} deg | "
    f"PDOP={best_pdop['pdop_mean']:.3f}"
)


print(
    f"Worst PDOP p95 : "
    f"{worst_pdop['scenario_id']} | "
    f"h={worst_pdop['altitude_km']:.0f} km | "
    f"i={worst_pdop['inclination_deg']:.1f} deg | "
    f"nu0={worst_pdop['initial_true_anomaly_deg']:.0f} deg | "
    f"PDOP95={worst_pdop['pdop_p95']:.3f}"
)


print()


# ------------------------------------------------------------
# 19. SCIENTIFIC FLAGS
# ------------------------------------------------------------

print(
    "----- SCIENTIFIC FLAGS -----"
)


print(
    f">=4 GNSS availability is 100% in all scenarios : "
    f"{all_ge4_full_availability}"
)


print(
    f">=6 AURORA redundancy availability is 100% in all scenarios : "
    f"{all_ge6_full_availability}"
)


print(
    f"Initial orbital phase measurably changes >=6 availability : "
    f"{phase_matters_for_ge6}"
)


print()


# ------------------------------------------------------------
# 20. FILES
# ------------------------------------------------------------

print(
    "----- FILES -----"
)


print(
    f"Scenario design : "
    f"{scenario_design_path}"
)


print(
    f"Scenario results : "
    f"{results_path}"
)


print(
    f"Altitude summary : "
    f"{altitude_summary_path}"
)


print(
    f"Inclination summary : "
    f"{inclination_summary_path}"
)


print(
    f"Worst geometries : "
    f"{worst_geometry_path}"
)


print()


# ------------------------------------------------------------
# 21. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)


print(
    f"Correct design scenario count : "
    f"{design_count_valid}"
)


print(
    f"Design IDs unique : "
    f"{design_ids_unique}"
)


print(
    f"All 108 scenarios complete : "
    f"{all_scenarios_complete}"
)


print(
    f"Scenario indices complete : "
    f"{indices_complete}"
)


print(
    f"All core metrics finite : "
    f"{all_core_metrics_finite}"
)


print(
    f"No PDOP geometry computation failures : "
    f"{no_geometry_failures}"
)


print(
    f"Availability bounds valid : "
    f"{availability_bounds_valid}"
)


print(
    f"Satellite-count ordering valid : "
    f"{satellite_count_order_valid}"
)


print(
    f"PDOP ordering valid : "
    f"{pdop_order_valid}"
)


print()


print(
    f"VALIDATION GLOBALE 017-A : "
    f"{validation_017a}"
)


print(
    "========================================================================================================================"
)


# ------------------------------------------------------------
# 22. FIGURE — MEAN PDOP VS ALTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


for inclination_deg in INCLINATIONS_DEG:

    subset = (
        results[
            np.isclose(
                results[
                    "inclination_deg"
                ],
                inclination_deg
            )
        ]
        .groupby(
            "altitude_km",
            as_index=False
        )
        .agg(
            pdop_mean=(
                "pdop_mean",
                "mean"
            )
        )
    )


    plt.plot(
        subset[
            "altitude_km"
        ],

        subset[
            "pdop_mean"
        ],

        marker="o",

        label=
            f"i={inclination_deg:.1f} deg"
    )


plt.xlabel(
    "Orbit altitude [km]"
)


plt.ylabel(
    "Mean PDOP"
)


plt.title(
    "AURORA — 017-A GNSS geometry vs altitude"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017a_pdop_vs_altitude.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 23. FIGURE — >=6 AVAILABILITY
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


for inclination_deg in INCLINATIONS_DEG:

    subset = (
        results[
            np.isclose(
                results[
                    "inclination_deg"
                ],
                inclination_deg
            )
        ]
        .groupby(
            "altitude_km",
            as_index=False
        )
        .agg(
            availability=(
                "availability_ge6_percent",
                "mean"
            )
        )
    )


    plt.plot(
        subset[
            "altitude_km"
        ],

        subset[
            "availability"
        ],

        marker="o",

        label=
            f"i={inclination_deg:.1f} deg"
    )


plt.xlabel(
    "Orbit altitude [km]"
)


plt.ylabel(
    "AURORA >=6 satellite availability [%]"
)


plt.title(
    "AURORA — 017-A integrity-redundancy geometry"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017a_integrity_availability.png",

    dpi=200
)


plt.show()


# ------------------------------------------------------------
# 24. FIGURE — PHASE DEPENDENCE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)


reference_subset = results[
    (
        np.isclose(
            results[
                "altitude_km"
            ],
            550.0
        )
    )
    &
    (
        np.isclose(
            results[
                "inclination_deg"
            ],
            97.6
        )
    )
].sort_values(
    "initial_true_anomaly_deg"
)


plt.plot(
    reference_subset[
        "initial_true_anomaly_deg"
    ],

    reference_subset[
        "pdop_mean"
    ],

    marker="o",

    label="Mean PDOP"
)


plt.xlabel(
    "Initial true anomaly [deg]"
)


plt.ylabel(
    "Mean PDOP"
)


plt.title(
    "AURORA — 017-A orbital-phase sensitivity at 550 km / 97.6 deg"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.savefig(
    figure_directory
    /
    "phase17_017a_reference_phase_pdop.png",

    dpi=200
)


plt.show()