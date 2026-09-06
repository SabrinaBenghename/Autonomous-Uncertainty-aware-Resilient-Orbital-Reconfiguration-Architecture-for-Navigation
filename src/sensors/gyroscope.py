import numpy as np


# ============================================================
# AURORA
# Gyroscope model
#
# The gyroscope measures:
#
#     omega_m,B
#         =
#     omega_true,B
#         +
#     b_g,B
#         +
#     n_g,B
#
#
# Units:
#
#     rad/s
# ============================================================


def simulate_gyroscope_measurement(
    true_angular_rate_body,
    bias_body,
    noise_std,
    rng
):
    """
    Simule un gyroscope trois axes.
    """

    true_angular_rate_body = np.asarray(
        true_angular_rate_body,
        dtype=float
    )


    bias_body = np.asarray(
        bias_body,
        dtype=float
    )


    noise_body = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=3
    )


    measurement_body = (
        true_angular_rate_body
        +
        bias_body
        +
        noise_body
    )


    return {
        "measurement":
            measurement_body,

        "true_angular_rate_body":
            true_angular_rate_body,

        "bias_body":
            bias_body,

        "noise_body":
            noise_body
    }