from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AURORA — EXPERIMENT 017-I
# EMPIRICAL RESILIENCE ENVELOPE
#
# ZERO NEW MISSIONS.
#
# This analysis combines:
#
#   017-F controlled estimator maturity
#   017-G mature-estimator outage-duration sweep
#   017-H maturity x duration stress test
#
# IMPORTANT:
#
# No absolute spacecraft requirement has been supplied.
# Therefore this script DOES NOT label conditions "safe" or
# "unsafe" and does not claim certification limits.
#
# Instead it constructs relative empirical degradation bands
# against the best mature short-outage baseline:
#
#   LOW       severity <= 2x baseline
#   MODERATE  severity <= 5x baseline
#   HIGH      severity > 5x baseline
#
# severity =
#   max(
#       mean outage RMSE / reference outage RMSE,
#       mean end error   / reference end error
#   )
#
# These are descriptive project-level bands only.
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

table_directory = Path("results") / "tables"
figure_directory = Path("results") / "figures"

f_path = (
    table_directory
    / "phase17_017f_controlled_estimator_maturity.csv"
)

g_path = (
    table_directory
    / "phase17_017g_outage_duration_sweep.csv"
)

h_path = (
    table_directory
    / "phase17_017h_combined_stress.csv"
)

output_conditions_path = (
    table_directory
    / "phase17_017i_empirical_resilience_conditions.csv"
)

output_envelope_path = (
    table_directory
    / "phase17_017i_empirical_resilience_envelope.csv"
)

output_evidence_path = (
    table_directory
    / "phase17_017i_evidence_summary.csv"
)

output_validation_path = (
    table_directory
    / "phase17_017i_validation.csv"
)


# ------------------------------------------------------------
# 2. REQUIRED INPUTS
# ------------------------------------------------------------

missing = [
    str(path)
    for path in [
        f_path,
        g_path,
        h_path,
    ]
    if not path.exists()
]

if missing:
    raise RuntimeError(
        "017-I aborted: required completed experiment files "
        "are missing:\n"
        +
        "\n".join(missing)
    )


f = pd.read_csv(f_path)
g = pd.read_csv(g_path)
h = pd.read_csv(h_path)


# ------------------------------------------------------------
# 3. NORMALIZE EACH CAMPAIGN TO AGE x DURATION CONDITIONS
# ------------------------------------------------------------

def summarize_conditions(
    dataframe,
    age_column,
    duration_column,
    source_label,
):
    required = [
        age_column,
        duration_column,
        "outage_rmse_m",
        "outage_end_error_m",
        "outage_start_sigma_m",
        "outage_end_sigma_m",
        "post_recovery_rmse_m",
    ]

    if not all(
        column in dataframe.columns
        for column in required
    ):
        raise RuntimeError(
            f"{source_label} does not contain expected columns."
        )

    temp = dataframe.copy()

    temp[
        "envelope_estimator_age_min"
    ] = temp[
        age_column
    ].astype(float)

    temp[
        "envelope_outage_duration_min"
    ] = temp[
        duration_column
    ].astype(float)

    summary = (
        temp.groupby(
            [
                "envelope_estimator_age_min",
                "envelope_outage_duration_min",
            ],
            as_index=False,
        )
        .agg(
            runs=(
                "replicate_index",
                "size",
            ),
            outage_start_sigma_mean_m=(
                "outage_start_sigma_m",
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
            outage_end_sigma_mean_m=(
                "outage_end_sigma_m",
                "mean",
            ),
            post_recovery_rmse_mean_m=(
                "post_recovery_rmse_m",
                "mean",
            ),
        )
    )

    summary[
        "source_experiment"
    ] = source_label

    return summary


# 017-F always used a 15-minute outage.
f_work = f.copy()

if (
    "experiment_outage_duration_min"
    not in f_work.columns
):
    f_work[
        "experiment_outage_duration_min"
    ] = 15.0

f_summary = summarize_conditions(
    dataframe=f_work,
    age_column=
        "experiment_estimator_age_min",
    duration_column=
        "experiment_outage_duration_min",
    source_label="017-F",
)


# 017-G always used a mature estimator started at t=0.
# At a 60-min outage start this is estimator age = 60 min.
g_work = g.copy()

g_work[
    "envelope_estimator_age_min"
] = 60.0

g_summary = summarize_conditions(
    dataframe=g_work.assign(
        experiment_estimator_age_min=60.0
    ),
    age_column=
        "experiment_estimator_age_min",
    duration_column=
        "experiment_outage_duration_min",
    source_label="017-G",
)


h_summary = summarize_conditions(
    dataframe=h,
    age_column=
        "experiment_estimator_age_min",
    duration_column=
        "experiment_outage_duration_min",
    source_label="017-H",
)


# ------------------------------------------------------------
# 4. MERGE OBSERVED CONDITION CELLS
#
# If the same age-duration point exists in multiple campaigns,
# retain all source summaries but also create one pooled
# envelope estimate weighted by number of runs.
# ------------------------------------------------------------

all_source_conditions = pd.concat(
    [
        f_summary,
        g_summary,
        h_summary,
    ],
    ignore_index=True,
)

all_source_conditions.to_csv(
    output_conditions_path,
    index=False,
)


def pooled_group(group):
    weights = group[
        "runs"
    ].to_numpy(dtype=float)

    def weighted_mean(column):
        values = group[
            column
        ].to_numpy(dtype=float)

        finite = np.isfinite(
            values
        )

        if not np.any(finite):
            return np.nan

        return float(
            np.average(
                values[finite],
                weights=weights[finite],
            )
        )

    return pd.Series(
        {
            "total_runs":
                int(
                    np.sum(weights)
                ),
            "source_count":
                int(
                    group[
                        "source_experiment"
                    ].nunique()
                ),
            "sources":
                ",".join(
                    sorted(
                        group[
                            "source_experiment"
                        ].unique()
                    )
                ),
            "outage_start_sigma_mean_m":
                weighted_mean(
                    "outage_start_sigma_mean_m"
                ),
            "outage_rmse_mean_m":
                weighted_mean(
                    "outage_rmse_mean_m"
                ),
            "outage_end_error_mean_m":
                weighted_mean(
                    "outage_end_error_mean_m"
                ),
            "outage_end_sigma_mean_m":
                weighted_mean(
                    "outage_end_sigma_mean_m"
                ),
            "post_recovery_rmse_mean_m":
                weighted_mean(
                    "post_recovery_rmse_mean_m"
                ),
        }
    )


envelope = (
    all_source_conditions.groupby(
        [
            "envelope_estimator_age_min",
            "envelope_outage_duration_min",
        ]
    )
    .apply(
        pooled_group
    )
    .reset_index()
)


# ------------------------------------------------------------
# 5. REFERENCE CONDITION
#
# Mature estimator (60 min) + shortest observed outage.
# ------------------------------------------------------------

mature_age = float(
    np.max(
        envelope[
            "envelope_estimator_age_min"
        ].to_numpy(dtype=float)
    )
)

mature_rows = envelope[
    np.isclose(
        envelope[
            "envelope_estimator_age_min"
        ].astype(float),
        mature_age,
    )
]

if len(
    mature_rows
) == 0:
    raise RuntimeError(
        "No mature-estimator condition available."
    )

reference_duration = float(
    np.min(
        mature_rows[
            "envelope_outage_duration_min"
        ].to_numpy(dtype=float)
    )
)

reference_row = mature_rows[
    np.isclose(
        mature_rows[
            "envelope_outage_duration_min"
        ].astype(float),
        reference_duration,
    )
].iloc[0]

reference_rmse = float(
    reference_row[
        "outage_rmse_mean_m"
    ]
)

reference_end_error = float(
    reference_row[
        "outage_end_error_mean_m"
    ]
)

if (
    not np.isfinite(reference_rmse)
    or
    not np.isfinite(reference_end_error)
    or
    reference_rmse <= 0.0
    or
    reference_end_error <= 0.0
):
    raise RuntimeError(
        "Invalid 017-I reference condition."
    )


# ------------------------------------------------------------
# 6. EMPIRICAL DEGRADATION INDEX
# ------------------------------------------------------------

LOW_BAND_MAX = 2.0
MODERATE_BAND_MAX = 5.0

envelope[
    "rmse_ratio_to_reference"
] = (
    envelope[
        "outage_rmse_mean_m"
    ]
    /
    reference_rmse
)

envelope[
    "end_error_ratio_to_reference"
] = (
    envelope[
        "outage_end_error_mean_m"
    ]
    /
    reference_end_error
)

envelope[
    "empirical_severity_index"
] = np.maximum(
    envelope[
        "rmse_ratio_to_reference"
    ].to_numpy(dtype=float),
    envelope[
        "end_error_ratio_to_reference"
    ].to_numpy(dtype=float),
)


def classify_severity(value):
    value = float(value)

    if value <= LOW_BAND_MAX:
        return "LOW_DEGRADATION"

    if value <= MODERATE_BAND_MAX:
        return "MODERATE_DEGRADATION"

    return "HIGH_DEGRADATION"


envelope[
    "empirical_degradation_band"
] = envelope[
    "empirical_severity_index"
].apply(
    classify_severity
)

envelope[
    "band_is_certification_limit"
] = False

envelope[
    "reference_estimator_age_min"
] = mature_age

envelope[
    "reference_outage_duration_min"
] = reference_duration

envelope[
    "reference_outage_rmse_m"
] = reference_rmse

envelope[
    "reference_end_error_m"
] = reference_end_error

envelope.to_csv(
    output_envelope_path,
    index=False,
)


# ------------------------------------------------------------
# 7. EVIDENCE SUMMARY
# ------------------------------------------------------------

evidence_rows = [
    {
        "experiment":
            "017-C",
        "question":
            "Does initial orbital phase materially change "
            "fixed-outage resilience?",
        "result":
            "No statistically significant phase dependence "
            "detected in the tested sweep.",
        "role_in_final_story":
            "Rules out initial phase as a dominant explanation "
            "for the tested fixed-outage case.",
    },
    {
        "experiment":
            "017-D",
        "question":
            "Does outage timing change resilience?",
        "result":
            "Strong timing dependence detected.",
        "role_in_final_story":
            "Reveals that when the outage occurs matters strongly.",
    },
    {
        "experiment":
            "017-E",
        "question":
            "Is timing dependence consistent with estimator maturity?",
        "result":
            "Strong observational consistency, but mission time "
            "and covariance were confounded.",
        "role_in_final_story":
            "Motivates controlled maturity intervention.",
    },
    {
        "experiment":
            "017-F",
        "question":
            "Does controlled estimator maturity affect resilience?",
        "result":
            "Yes; maturity significantly changes outage RMSE and "
            "end-of-outage error at fixed outage timing.",
        "role_in_final_story":
            "Confirms estimator maturity as an important mechanism.",
    },
    {
        "experiment":
            "017-G",
        "question":
            "How does outage duration affect a mature estimator?",
        "result":
            "Loaded from completed 017-G campaign.",
        "role_in_final_story":
            "Defines duration-dependent degradation.",
    },
    {
        "experiment":
            "017-H",
        "question":
            "How do estimator maturity and long outage duration "
            "combine under stress?",
        "result":
            "Loaded from completed 017-H campaign.",
        "role_in_final_story":
            "Tests combined stress and interaction-like behavior.",
    },
    {
        "experiment":
            "017-I",
        "question":
            "What empirical resilience envelope is supported by "
            "the completed evidence?",
        "result":
            "Observed conditions classified by relative empirical "
            "degradation; no absolute safety certification claimed.",
        "role_in_final_story":
            "Produces final project-level resilience map.",
    },
]

evidence = pd.DataFrame(
    evidence_rows
)

evidence.to_csv(
    output_evidence_path,
    index=False,
)


# ------------------------------------------------------------
# 8. VALIDATION
# ------------------------------------------------------------

required_envelope_columns = [
    "envelope_estimator_age_min",
    "envelope_outage_duration_min",
    "outage_rmse_mean_m",
    "outage_end_error_mean_m",
    "empirical_severity_index",
    "empirical_degradation_band",
]

inputs_nonempty = bool(
    len(f) > 0
    and
    len(g) > 0
    and
    len(h) > 0
)

envelope_nonempty = bool(
    len(envelope) > 0
)

required_columns_present = bool(
    all(
        column in envelope.columns
        for column in required_envelope_columns
    )
)

primary_metrics_finite = bool(
    np.all(
        np.isfinite(
            envelope[
                [
                    "outage_rmse_mean_m",
                    "outage_end_error_mean_m",
                    "empirical_severity_index",
                ]
            ].to_numpy(dtype=float)
        )
    )
)

reference_valid = bool(
    np.isfinite(reference_rmse)
    and
    np.isfinite(reference_end_error)
    and
    reference_rmse > 0.0
    and
    reference_end_error > 0.0
)

contains_multiple_ages = bool(
    envelope[
        "envelope_estimator_age_min"
    ].nunique()
    >=
    3
)

contains_multiple_durations = bool(
    envelope[
        "envelope_outage_duration_min"
    ].nunique()
    >=
    3
)

validation_017i = bool(
    inputs_nonempty
    and
    envelope_nonempty
    and
    required_columns_present
    and
    primary_metrics_finite
    and
    reference_valid
    and
    contains_multiple_ages
    and
    contains_multiple_durations
)

pd.DataFrame(
    [
        {
            "inputs_nonempty":
                inputs_nonempty,
            "envelope_nonempty":
                envelope_nonempty,
            "required_columns_present":
                required_columns_present,
            "primary_metrics_finite":
                primary_metrics_finite,
            "reference_valid":
                reference_valid,
            "contains_multiple_ages":
                contains_multiple_ages,
            "contains_multiple_durations":
                contains_multiple_durations,
            "global_validation":
                validation_017i,
        }
    ]
).to_csv(
    output_validation_path,
    index=False,
)


# ------------------------------------------------------------
# 9. PRINT
# ------------------------------------------------------------

print()
print("=" * 120)
print("AURORA — EXPERIMENT 017-I")
print("Empirical resilience envelope — ZERO NEW MISSIONS")
print("=" * 120)
print(
    f"017-F missions loaded : "
    f"{len(f)}"
)
print(
    f"017-G missions loaded : "
    f"{len(g)}"
)
print(
    f"017-H missions loaded : "
    f"{len(h)}"
)
print()

print("----- REFERENCE CONDITION -----")
print(
    f"Estimator age : "
    f"{mature_age:.1f} min"
)
print(
    f"Outage duration : "
    f"{reference_duration:.1f} min"
)
print(
    f"Reference mean outage RMSE : "
    f"{reference_rmse:.4f} m"
)
print(
    f"Reference mean end error : "
    f"{reference_end_error:.4f} m"
)
print()

print("----- EMPIRICAL RESILIENCE ENVELOPE -----")
display_columns = [
    "envelope_estimator_age_min",
    "envelope_outage_duration_min",
    "total_runs",
    "outage_start_sigma_mean_m",
    "outage_rmse_mean_m",
    "outage_end_error_mean_m",
    "empirical_severity_index",
    "empirical_degradation_band",
    "sources",
]

print(
    envelope[
        display_columns
    ]
    .sort_values(
        [
            "envelope_estimator_age_min",
            "envelope_outage_duration_min",
        ]
    )
    .to_string(
        index=False,
        formatters={
            "envelope_estimator_age_min":
                lambda value:
                    f"{value:.1f}",
            "envelope_outage_duration_min":
                lambda value:
                    f"{value:.1f}",
            "outage_start_sigma_mean_m":
                lambda value:
                    f"{value:.4f}",
            "outage_rmse_mean_m":
                lambda value:
                    f"{value:.4f}",
            "outage_end_error_mean_m":
                lambda value:
                    f"{value:.4f}",
            "empirical_severity_index":
                lambda value:
                    f"{value:.2f}",
        },
    )
)
print()

print("----- BAND DEFINITION -----")
print(
    "LOW_DEGRADATION      : severity <= 2x baseline"
)
print(
    "MODERATE_DEGRADATION : 2x < severity <= 5x baseline"
)
print(
    "HIGH_DEGRADATION     : severity > 5x baseline"
)
print(
    "These are empirical relative bands, NOT spacecraft "
    "certification or mission-safety limits."
)
print()

print("----- VALIDATION -----")
print(
    f"All required experiment inputs present : "
    f"{inputs_nonempty}"
)
print(
    f"Envelope non-empty : "
    f"{envelope_nonempty}"
)
print(
    f"Primary envelope metrics finite : "
    f"{primary_metrics_finite}"
)
print(
    f"Reference condition valid : "
    f"{reference_valid}"
)
print(
    f"Multiple estimator ages represented : "
    f"{contains_multiple_ages}"
)
print(
    f"Multiple outage durations represented : "
    f"{contains_multiple_durations}"
)
print()
print(
    f"VALIDATION GLOBALE 017-I : "
    f"{validation_017i}"
)
print("=" * 120)


# ------------------------------------------------------------
# 10. FIGURES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for age_min in sorted(
    envelope[
        "envelope_estimator_age_min"
    ].unique()
):
    block = envelope[
        np.isclose(
            envelope[
                "envelope_estimator_age_min"
            ].astype(float),
            age_min,
        )
    ].sort_values(
        "envelope_outage_duration_min"
    )

    plt.plot(
        block[
            "envelope_outage_duration_min"
        ],
        block[
            "outage_rmse_mean_m"
        ],
        marker="o",
        label=
            f"Estimator age {age_min:.0f} min",
    )

plt.xlabel("Dual-outage duration [min]")
plt.ylabel("Mean outage RMSE [m]")
plt.title(
    "AURORA — 017-I empirical resilience envelope"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017i_resilience_envelope_rmse.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(10, 6))

for age_min in sorted(
    envelope[
        "envelope_estimator_age_min"
    ].unique()
):
    block = envelope[
        np.isclose(
            envelope[
                "envelope_estimator_age_min"
            ].astype(float),
            age_min,
        )
    ].sort_values(
        "envelope_outage_duration_min"
    )

    plt.plot(
        block[
            "envelope_outage_duration_min"
        ],
        block[
            "empirical_severity_index"
        ],
        marker="o",
        label=
            f"Estimator age {age_min:.0f} min",
    )

plt.axhline(
    LOW_BAND_MAX,
    linestyle="--",
)
plt.axhline(
    MODERATE_BAND_MAX,
    linestyle="--",
)
plt.xlabel("Dual-outage duration [min]")
plt.ylabel("Relative empirical severity index")
plt.title(
    "AURORA — 017-I relative degradation envelope"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    figure_directory
    / "phase17_017i_relative_degradation_envelope.png",
    dpi=200,
)
plt.close()
