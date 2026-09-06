import numpy as np


# ============================================================
# AURORA
# Modèle d'accéléromètre
# ============================================================


def simulate_accelerometer_measurement(
    true_specific_force_eci,
    bias_eci,
    noise_std,
    rng
):
    """
    Simule une mesure d'accéléromètre.

    Modèle :

        f_meas = f_true + b_a + noise

    IMPORTANT :
    On suppose ici que la mesure a déjà été
    parfaitement transformée du repère capteur
    vers ECI.

    L'attitude et les gyroscopes seront ajoutés
    dans une phase ultérieure.
    """

    true_specific_force_eci = np.asarray(
        true_specific_force_eci,
        dtype=float
    )

    bias_eci = np.asarray(
        bias_eci,
        dtype=float
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=3
    )

    measurement = (
        true_specific_force_eci
        + bias_eci
        + noise
    )

    return {
        "measurement":
            measurement,

        "true_specific_force":
            true_specific_force_eci,

        "bias":
            bias_eci,

        "noise":
            noise
    }