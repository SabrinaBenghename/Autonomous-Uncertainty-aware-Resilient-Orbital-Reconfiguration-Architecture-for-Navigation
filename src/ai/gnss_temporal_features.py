import numpy as np
import pandas as pd


# ============================================================
# AURORA
# Phase 15-C
#
# Causal temporal GNSS anomaly features
#
#
# IMPORTANT:
#
# All temporal features use only:
#
#       x_k
#       x_{k-1}
#       x_{k-2}
#       ...
#
# NEVER future samples.
#
# Therefore they are compatible with an onboard
# real-time implementation.
# ============================================================


TEMPORAL_SOURCE_FEATURES = [
    "chi2_ratio",
    "residual_max_abs_m",
    "prior_position_correction_m"
]


TEMPORAL_WINDOWS = [
    3,
    6,
    12
]


def build_temporal_feature_columns():
    """
    Return the complete list of features used by the
    temporal anomaly detector.
    """

    columns = [
        "pdop"
    ]

    # --------------------------------------------------------
    # Current instantaneous features
    # --------------------------------------------------------

    columns.extend(
        TEMPORAL_SOURCE_FEATURES
    )


    # --------------------------------------------------------
    # Causal rolling means
    # --------------------------------------------------------

    for feature_name in TEMPORAL_SOURCE_FEATURES:

        for window in TEMPORAL_WINDOWS:

            columns.append(
                f"{feature_name}_mean_{window}"
            )


    # --------------------------------------------------------
    # Causal rolling maxima
    #
    # Only medium / long windows.
    # --------------------------------------------------------

    for feature_name in TEMPORAL_SOURCE_FEATURES:

        for window in [
            6,
            12
        ]:

            columns.append(
                f"{feature_name}_max_{window}"
            )


    # --------------------------------------------------------
    # One-step temporal derivative
    # --------------------------------------------------------

    for feature_name in TEMPORAL_SOURCE_FEATURES:

        columns.append(
            f"{feature_name}_delta_1"
        )


    return columns


TEMPORAL_FEATURE_COLUMNS = (
    build_temporal_feature_columns()
)


def add_causal_temporal_features(
    dataframe,
    group_columns,
    time_column="time_seconds"
):
    """
    Add causal rolling features to a GNSS feature dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    group_columns : list[str]
        Columns defining independent temporal sequences.

        Example:
            ["run_id", "fault_magnitude_m"]

    time_column : str
        Temporal sorting column.

    Returns
    -------
    pandas.DataFrame
    """

    result = dataframe.copy()


    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        list(
            group_columns
        )
        +
        [
            time_column,
            "pdop"
        ]
        +
        TEMPORAL_SOURCE_FEATURES
    )


    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]


    if len(
        missing_columns
    ) > 0:

        raise ValueError(
            "Colonnes manquantes pour features temporelles : "
            +
            str(
                missing_columns
            )
        )


    # --------------------------------------------------------
    # Sort sequences
    # --------------------------------------------------------

    result = result.sort_values(
        by=
            list(
                group_columns
            )
            +
            [
                time_column
            ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    for feature_name in TEMPORAL_SOURCE_FEATURES:

        grouped_feature = (
            result.groupby(
                group_columns,
                sort=False
            )[
                feature_name
            ]
        )


        # ====================================================
        # Rolling means
        # ====================================================

        for window in TEMPORAL_WINDOWS:

            result[
                f"{feature_name}_mean_{window}"
            ] = (
                grouped_feature.transform(
                    lambda values:
                        values.rolling(
                            window=
                                window,

                            min_periods=
                                window
                        ).mean()
                )
            )


        # ====================================================
        # Rolling maxima
        # ====================================================

        for window in [
            6,
            12
        ]:

            result[
                f"{feature_name}_max_{window}"
            ] = (
                grouped_feature.transform(
                    lambda values:
                        values.rolling(
                            window=
                                window,

                            min_periods=
                                window
                        ).max()
                )
            )


        # ====================================================
        # One-step derivative
        # ====================================================

        result[
            f"{feature_name}_delta_1"
        ] = (
            grouped_feature.diff(
                periods=1
            )
        )


    return result


def temporal_feature_valid_mask(
    dataframe
):
    """
    Return True only for rows where every temporal feature
    is available and finite.
    """

    feature_matrix = dataframe[
        TEMPORAL_FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )


    return np.all(
        np.isfinite(
            feature_matrix
        ),
        axis=1
    )