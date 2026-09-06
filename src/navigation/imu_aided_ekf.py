import numpy as np

from src.navigation.orbit_ekf_models import (
    j2_state_derivative,
    ekf_predict_variational,
    build_discrete_acceleration_process_noise
)


# ============================================================
# AURORA
# EKF orbital aidé par accéléromètre
# ============================================================


def build_j2_accelerometer_dynamics(
    measured_specific_force_eci
):
    """
    Construit une dynamique :

        gravity + J2 + acceleration mesuree

    La mesure est considérée constante
    pendant un pas temporel.
    """

    measured_specific_force_eci = (
        np.asarray(
            measured_specific_force_eci,
            dtype=float
        )
    )


    def dynamics_function(
        state
    ):

        derivative = (
            j2_state_derivative(
                state
            ).copy()
        )

        derivative[
            3:6
        ] += measured_specific_force_eci

        return derivative


    return dynamics_function


def ekf_predict_with_accelerometer(
    state,
    covariance,
    dt,
    accelerometer_measurement_eci,
    accelerometer_noise_std
):
    """
    Prediction EKF utilisant :

        - gravité centrale
        - J2
        - accéléromètre

    Le bruit de l'accéléromètre est propagé
    dans la covariance via Q.
    """

    dynamics_function = (
        build_j2_accelerometer_dynamics(
            accelerometer_measurement_eci
        )
    )

    process_noise_covariance = (
        build_discrete_acceleration_process_noise(
            dt=
                dt,

            acceleration_sigma=
                accelerometer_noise_std
        )
    )

    return ekf_predict_variational(
        state=
            state,

        covariance=
            covariance,

        dt=
            dt,

        process_noise_covariance=
            process_noise_covariance,

        dynamics_function=
            dynamics_function
    )