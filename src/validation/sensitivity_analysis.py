import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr
)

from sklearn.linear_model import (
    RidgeCV
)

from sklearn.model_selection import (
    KFold,
    cross_val_score
)

from sklearn.pipeline import (
    make_pipeline
)

from sklearn.preprocessing import (
    StandardScaler
)


# ============================================================
# AURORA
# Phase 16-D
#
# Sensitivity-screening utilities
#
#
# PRIMARY METHOD
#
#       Spearman rank correlation
#
# Why:
#
#   - interpretable
#   - detects monotonic nonlinear relationships
#   - does not assume Gaussian variables
#   - suitable for Latin-Hypercube screening
#
#
# SECONDARY METHOD
#
#       Standardized Ridge regression
#
# Why:
#
#   - multivariate
#   - partially accounts for simultaneous parameter changes
#   - standardized coefficients are comparable
#
#
# IMPORTANT:
#
# This is a SCREENING sensitivity analysis.
#
# It is NOT:
#
#   - a Sobol global sensitivity decomposition
#   - a causal analysis
#   - proof that a parameter causes an observed metric
#
# With N = 64 and D = 23, rankings must be interpreted
# as evidence about the tested design domain.
# ============================================================


# ------------------------------------------------------------
# 1. ORIGINAL PHASE-16 DESIGN VARIABLES
# ------------------------------------------------------------

PHASE16_INPUT_PARAMETERS = [

    "pseudorange_noise_std_m",

    "accelerometer_noise_std_mps2",

    "accelerometer_bias_x_mps2",
    "accelerometer_bias_y_mps2",
    "accelerometer_bias_z_mps2",

    "gyro_noise_std_degps",

    "gyro_bias_x_degps",
    "gyro_bias_y_degps",
    "gyro_bias_z_degps",

    "star_tracker_noise_std_deg",

    "initial_position_error_x_m",
    "initial_position_error_y_m",
    "initial_position_error_z_m",

    "initial_velocity_error_x_mps",
    "initial_velocity_error_y_mps",
    "initial_velocity_error_z_mps",

    "non_gravitational_acceleration_mps2",

    "fault_magnitude_m",
    "fault_start_min",
    "fault_duration_min",

    "gnss_outage_duration_min",
    "star_tracker_outage_duration_min",
    "dual_outage_duration_min"
]


# ------------------------------------------------------------
# 2. PHYSICALLY MOTIVATED DERIVED DIAGNOSTICS
#
# These are NOT additional independent design variables.
#
# They are only used for secondary physical interpretation.
# ------------------------------------------------------------

DERIVED_SENSITIVITY_PREDICTORS = [

    "accelerometer_bias_norm_mps2",

    "gyro_bias_norm_degps",

    "initial_position_error_norm_m",

    "initial_velocity_error_norm_mps",

    "fault_to_noise_ratio"
]


# ------------------------------------------------------------
# 3. SUBSYSTEM CLASSIFICATION
# ------------------------------------------------------------

PARAMETER_SUBSYSTEM = {

    "pseudorange_noise_std_m":
        "GNSS measurement",

    "accelerometer_noise_std_mps2":
        "Accelerometer",

    "accelerometer_bias_x_mps2":
        "Accelerometer",

    "accelerometer_bias_y_mps2":
        "Accelerometer",

    "accelerometer_bias_z_mps2":
        "Accelerometer",

    "gyro_noise_std_degps":
        "Gyroscope",

    "gyro_bias_x_degps":
        "Gyroscope",

    "gyro_bias_y_degps":
        "Gyroscope",

    "gyro_bias_z_degps":
        "Gyroscope",

    "star_tracker_noise_std_deg":
        "Star tracker",

    "initial_position_error_x_m":
        "Initial state",

    "initial_position_error_y_m":
        "Initial state",

    "initial_position_error_z_m":
        "Initial state",

    "initial_velocity_error_x_mps":
        "Initial state",

    "initial_velocity_error_y_mps":
        "Initial state",

    "initial_velocity_error_z_mps":
        "Initial state",

    "non_gravitational_acceleration_mps2":
        "Orbital disturbance",

    "fault_magnitude_m":
        "GNSS fault",

    "fault_start_min":
        "GNSS fault",

    "fault_duration_min":
        "GNSS fault",

    "gnss_outage_duration_min":
        "Mission outage",

    "star_tracker_outage_duration_min":
        "Mission outage",

    "dual_outage_duration_min":
        "Mission outage"
}


# ------------------------------------------------------------
# 4. ADD DERIVED PHYSICAL VARIABLES
# ------------------------------------------------------------

def add_derived_sensitivity_predictors(
    dataframe
):

    dataframe = dataframe.copy()


    dataframe[
        "accelerometer_bias_norm_mps2"
    ] = np.sqrt(
        dataframe[
            "accelerometer_bias_x_mps2"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "accelerometer_bias_y_mps2"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "accelerometer_bias_z_mps2"
        ].to_numpy(
            dtype=float
        )**2
    )


    dataframe[
        "gyro_bias_norm_degps"
    ] = np.sqrt(
        dataframe[
            "gyro_bias_x_degps"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "gyro_bias_y_degps"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "gyro_bias_z_degps"
        ].to_numpy(
            dtype=float
        )**2
    )


    dataframe[
        "initial_position_error_norm_m"
    ] = np.sqrt(
        dataframe[
            "initial_position_error_x_m"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "initial_position_error_y_m"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "initial_position_error_z_m"
        ].to_numpy(
            dtype=float
        )**2
    )


    dataframe[
        "initial_velocity_error_norm_mps"
    ] = np.sqrt(
        dataframe[
            "initial_velocity_error_x_mps"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "initial_velocity_error_y_mps"
        ].to_numpy(
            dtype=float
        )**2
        +
        dataframe[
            "initial_velocity_error_z_mps"
        ].to_numpy(
            dtype=float
        )**2
    )


    pseudorange_noise = (
        dataframe[
            "pseudorange_noise_std_m"
        ].to_numpy(
            dtype=float
        )
    )


    dataframe[
        "fault_to_noise_ratio"
    ] = (
        dataframe[
            "fault_magnitude_m"
        ].to_numpy(
            dtype=float
        )
        /
        pseudorange_noise
    )


    return dataframe


# ------------------------------------------------------------
# 5. BENJAMINI-HOCHBERG FDR CORRECTION
# ------------------------------------------------------------

def benjamini_hochberg(
    p_values
):
    """
    Convert p-values into Benjamini-Hochberg q-values.

    Correction is intended to be applied within one output
    metric across the tested input parameters.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )


    q_values = np.full(
        p_values.shape,
        np.nan,
        dtype=float
    )


    finite_mask = np.isfinite(
        p_values
    )


    finite_indices = np.where(
        finite_mask
    )[0]


    if len(
        finite_indices
    ) == 0:

        return q_values


    finite_p_values = (
        p_values[
            finite_indices
        ]
    )


    ordering = np.argsort(
        finite_p_values
    )


    sorted_p_values = (
        finite_p_values[
            ordering
        ]
    )


    number_of_tests = len(
        sorted_p_values
    )


    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float
    )


    adjusted = (
        sorted_p_values
        *
        number_of_tests
        /
        ranks
    )


    # ========================================================
    # Enforce monotonic q-values from largest p-value backward.
    # ========================================================

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]


    adjusted = np.clip(
        adjusted,
        0.0,
        1.0
    )


    inverse_order = np.empty_like(
        ordering
    )


    inverse_order[
        ordering
    ] = np.arange(
        number_of_tests
    )


    finite_q_values = (
        adjusted[
            inverse_order
        ]
    )


    q_values[
        finite_indices
    ] = (
        finite_q_values
    )


    return q_values


# ------------------------------------------------------------
# 6. SPEARMAN SCREENING
# ------------------------------------------------------------

def compute_spearman_screening(
    dataframe,
    input_columns,
    output_columns
):

    rows = []


    for output_name in output_columns:

        temporary_rows = []


        for input_name in input_columns:

            x = dataframe[
                input_name
            ].to_numpy(
                dtype=float
            )


            y = dataframe[
                output_name
            ].to_numpy(
                dtype=float
            )


            valid = (
                np.isfinite(
                    x
                )
                &
                np.isfinite(
                    y
                )
            )


            x_valid = (
                x[
                    valid
                ]
            )


            y_valid = (
                y[
                    valid
                ]
            )


            if (
                len(
                    x_valid
                )
                <
                4
                or
                np.std(
                    x_valid
                )
                ==
                0.0
                or
                np.std(
                    y_valid
                )
                ==
                0.0
            ):

                correlation = np.nan

                p_value = np.nan


            else:

                correlation, p_value = (
                    spearmanr(
                        x_valid,
                        y_valid
                    )
                )


            temporary_rows.append(
                {
                    "output_metric":
                        output_name,

                    "parameter":
                        input_name,

                    "spearman_rho":
                        float(
                            correlation
                        ),

                    "absolute_rho":
                        float(
                            abs(
                                correlation
                            )
                        ),

                    "p_value":
                        float(
                            p_value
                        ),

                    "samples":
                        int(
                            len(
                                x_valid
                            )
                        )
                }
            )


        temporary_dataframe = pd.DataFrame(
            temporary_rows
        )


        temporary_dataframe[
            "q_value"
        ] = (
            benjamini_hochberg(
                temporary_dataframe[
                    "p_value"
                ].to_numpy(
                    dtype=float
                )
            )
        )


        temporary_dataframe[
            "fdr_significant_0p10"
        ] = (
            temporary_dataframe[
                "q_value"
            ]
            <=
            0.10
        )


        temporary_dataframe[
            "rank_abs_rho"
        ] = (
            temporary_dataframe[
                "absolute_rho"
            ]
            .rank(
                method="min",
                ascending=False
            )
            .astype(
                int
            )
        )


        rows.append(
            temporary_dataframe
        )


    result = pd.concat(
        rows,
        ignore_index=True
    )


    return result


# ------------------------------------------------------------
# 7. STANDARDIZED RIDGE SCREENING
# ------------------------------------------------------------

def compute_standardized_ridge_screening(
    dataframe,
    input_columns,
    output_columns,
    random_state=16004
):
    """
    Secondary multivariate screening.

    X and y are standardized before the final fit, therefore
    coefficient magnitudes are dimensionless and comparable
    inside each output metric.
    """

    coefficient_rows = []

    model_rows = []


    alpha_grid = np.logspace(
        -3,
        3,
        61
    )


    number_of_samples = len(
        dataframe
    )


    outer_folds = min(
        5,
        number_of_samples
    )


    outer_cv = KFold(
        n_splits=
            outer_folds,

        shuffle=
            True,

        random_state=
            random_state
    )


    for output_name in output_columns:

        X = dataframe[
            input_columns
        ].to_numpy(
            dtype=float
        )


        y = dataframe[
            output_name
        ].to_numpy(
            dtype=float
        )


        valid = (
            np.all(
                np.isfinite(
                    X
                ),
                axis=1
            )
            &
            np.isfinite(
                y
            )
        )


        X = (
            X[
                valid
            ]
        )


        y = (
            y[
                valid
            ]
        )


        if len(
            y
        ) < 10:

            raise ValueError(
                f"Not enough samples for Ridge: {output_name}"
            )


        # ====================================================
        # Outer CV diagnostic
        # ====================================================

        pipeline = make_pipeline(
            StandardScaler(),

            RidgeCV(
                alphas=
                    alpha_grid,

                cv=
                    5
            )
        )


        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=
                outer_cv,
            scoring=
                "r2"
        )


        # ====================================================
        # Final standardized model for coefficient ranking
        # ====================================================

        x_scaler = StandardScaler()


        X_standardized = (
            x_scaler.fit_transform(
                X
            )
        )


        y_mean = np.mean(
            y
        )


        y_std = np.std(
            y,
            ddof=0
        )


        if y_std <= 1.0e-15:

            y_standardized = np.zeros_like(
                y
            )

        else:

            y_standardized = (
                y
                -
                y_mean
            ) / y_std


        ridge = RidgeCV(
            alphas=
                alpha_grid,

            cv=
                5
        )


        ridge.fit(
            X_standardized,
            y_standardized
        )


        coefficients = np.asarray(
            ridge.coef_,
            dtype=float
        )


        absolute_coefficients = np.abs(
            coefficients
        )


        coefficient_ranks = (
            pd.Series(
                absolute_coefficients
            )
            .rank(
                method="min",
                ascending=False
            )
            .astype(
                int
            )
            .to_numpy()
        )


        for parameter_index, parameter_name in enumerate(
            input_columns
        ):

            coefficient_rows.append(
                {
                    "output_metric":
                        output_name,

                    "parameter":
                        parameter_name,

                    "standardized_coefficient":
                        float(
                            coefficients[
                                parameter_index
                            ]
                        ),

                    "absolute_coefficient":
                        float(
                            absolute_coefficients[
                                parameter_index
                            ]
                        ),

                    "rank_abs_coefficient":
                        int(
                            coefficient_ranks[
                                parameter_index
                            ]
                        )
                }
            )


        model_rows.append(
            {
                "output_metric":
                    output_name,

                "selected_alpha":
                    float(
                        ridge.alpha_
                    ),

                "training_r2":
                    float(
                        ridge.score(
                            X_standardized,
                            y_standardized
                        )
                    ),

                "cv_r2_mean":
                    float(
                        np.mean(
                            cv_scores
                        )
                    ),

                "cv_r2_std":
                    float(
                        np.std(
                            cv_scores,
                            ddof=1
                        )
                    )
            }
        )


    coefficient_table = pd.DataFrame(
        coefficient_rows
    )


    model_table = pd.DataFrame(
        model_rows
    )


    return (
        coefficient_table,
        model_table
    )


# ------------------------------------------------------------
# 8. BUILD TOP-FACTOR TABLE
# ------------------------------------------------------------

def build_top_factor_table(
    spearman_table,
    top_n=5
):

    rows = []


    for output_metric in (
        spearman_table[
            "output_metric"
        ].unique()
    ):

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
                top_n
            )
        )


        for rank_index, (_, row) in enumerate(
            subset.iterrows(),
            start=1
        ):

            rows.append(
                {
                    "output_metric":
                        output_metric,

                    "rank":
                        rank_index,

                    "parameter":
                        row[
                            "parameter"
                        ],

                    "spearman_rho":
                        row[
                            "spearman_rho"
                        ],

                    "absolute_rho":
                        row[
                            "absolute_rho"
                        ],

                    "p_value":
                        row[
                            "p_value"
                        ],

                    "q_value":
                        row[
                            "q_value"
                        ],

                    "fdr_significant_0p10":
                        row[
                            "fdr_significant_0p10"
                        ]
                }
            )


    return pd.DataFrame(
        rows
    )