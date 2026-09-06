import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    synthetic_drag_like_acceleration,
    propagate_orbit_with_j2_and_drag_like
)

from src.navigation.orbit_ekf import (
    ekf_update_position
)

from src.navigation.imu_aided_ekf import (
    ekf_predict_with_accelerometer
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_predict_with_bias_estimation,
    ekf_update_position_augmented
)

from src.sensors.accelerometer import (
    simulate_accelerometer_measurement
)


# ============================================================
# AURORA
# Expérience 008-B
#
# Estimation du biais accéléromètre
#
# Comparaison :
#
# 1. EKF 6D
#    utilise l'accéléromètre mais ignore son biais
#
# 2. EKF 9D
#    estime position, vitesse ET biais
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH + 550_000.0
)

eccentricity = (
    0.01
)

inclination = np.deg2rad(
    97.6
)

raan = np.deg2rad(
    40.0
)

argument_of_periapsis = np.deg2rad(
    30.0
)

true_anomaly = np.deg2rad(
    25.0
)


# ------------------------------------------------------------
# 2. TEMPS
# ------------------------------------------------------------

simulation_duration = (
    60.0 * 60.0
)

desired_dt = (
    10.0
)


number_of_epochs = (
    int(
        np.floor(
            simulation_duration
            / desired_dt
        )
    )
    + 1
)


time = np.linspace(
    0.0,
    simulation_duration,
    number_of_epochs
)


dt = (
    time[1]
    - time[0]
)


time_minutes = (
    time / 60.0
)


# ------------------------------------------------------------
# 3. COUPURE GNSS
# ------------------------------------------------------------

gnss_outage_start_minutes = (
    20.0
)

gnss_outage_end_minutes = (
    40.0
)


gnss_available = ~(
    (
        time_minutes
        >= gnss_outage_start_minutes
    )
    &
    (
        time_minutes
        <= gnss_outage_end_minutes
    )
)


outage_mask = (
    ~gnss_available
)


# ------------------------------------------------------------
# 4. FORCE NON GRAVITATIONNELLE
# ------------------------------------------------------------

true_non_gravitational_acceleration = (
    2.0e-5
)


# ------------------------------------------------------------
# 5. ETAT INITIAL VRAI
# ------------------------------------------------------------

true_initial_position, true_initial_velocity = (
    keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )
)


true_initial_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)


# ------------------------------------------------------------
# 6. VERITE
# ------------------------------------------------------------

truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            true_non_gravitational_acceleration
    )
)


truth_states = (
    truth_solution.y.T
)


maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        - time
    )
)


# ------------------------------------------------------------
# 7. MESURES GNSS
# ------------------------------------------------------------

gnss_position_sigma = (
    3.0
)


R = (
    gnss_position_sigma**2
    * np.eye(3)
)


gnss_rng = np.random.default_rng(
    2026
)


gnss_measurements = (
    truth_states[:, 0:3]
    +
    gnss_rng.normal(
        loc=0.0,
        scale=gnss_position_sigma,
        size=(
            number_of_epochs,
            3
        )
    )
)


# ------------------------------------------------------------
# 8. ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


true_accelerometer_bias = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])


accelerometer_rng = (
    np.random.default_rng(
        4242
    )
)


true_specific_forces = np.zeros(
    (
        number_of_epochs,
        3
    )
)


accelerometer_measurements = np.zeros(
    (
        number_of_epochs,
        3
    )
)


for index in range(
    number_of_epochs
):

    true_specific_force = (
        synthetic_drag_like_acceleration(
            time=
                time[
                    index
                ],

            position=
                truth_states[
                    index,
                    0:3
                ],

            velocity=
                truth_states[
                    index,
                    3:6
                ],

            base_acceleration=
                true_non_gravitational_acceleration
        )
    )


    measurement_result = (
        simulate_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force,

            bias_eci=
                true_accelerometer_bias,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    true_specific_forces[
        index
    ] = (
        true_specific_force
    )


    accelerometer_measurements[
        index
    ] = (
        measurement_result[
            "measurement"
        ]
    )


# ------------------------------------------------------------
# 9. INITIALISATION ORBITALE
# ------------------------------------------------------------

initial_position_error = np.array([
    100.0,
    -80.0,
    60.0
])


initial_velocity_error = np.array([
    0.10,
    -0.08,
    0.05
])


estimated_initial_orbital_state = (
    true_initial_state
    +
    np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )
)


# ------------------------------------------------------------
# 10. COVARIANCE INITIALE 6D
# ------------------------------------------------------------

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)


P0_6 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2
])


# ------------------------------------------------------------
# 11. COVARIANCE INITIALE 9D
# ------------------------------------------------------------

initial_bias_estimate = np.zeros(
    3
)


initial_bias_sigma = (
    5.0e-6
)


estimated_initial_augmented_state = (
    np.concatenate(
        (
            estimated_initial_orbital_state,
            initial_bias_estimate
        )
    )
)


P0_9 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2,

    initial_bias_sigma**2,
    initial_bias_sigma**2,
    initial_bias_sigma**2
])


# ------------------------------------------------------------
# 12. MARCHE ALEATOIRE DU BIAIS
# ------------------------------------------------------------

bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 13. FILTRE 6D :
# ACCELEROMETRE SANS ESTIMATION DE BIAIS
# ------------------------------------------------------------

def run_filter_without_bias_estimation():

    estimated_state = (
        estimated_initial_orbital_state.copy()
    )


    covariance = (
        P0_6.copy()
    )


    position_errors = np.zeros(
        number_of_epochs
    )


    position_sigmas = np.zeros(
        number_of_epochs
    )


    nees_values = np.full(
        number_of_epochs,
        np.nan
    )


    nis_values = np.full(
        number_of_epochs,
        np.nan
    )


    initial_error = (
        estimated_state
        - truth_states[0]
    )


    position_errors[0] = np.linalg.norm(
        initial_error[
            0:3
        ]
    )


    position_sigmas[0] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    nees_values[0] = (
        initial_error.T
        @ np.linalg.solve(
            covariance,
            initial_error
        )
    )


    for index in range(
        1,
        number_of_epochs
    ):

        (
            predicted_state,
            predicted_covariance
        ) = ekf_predict_with_accelerometer(
            state=
                estimated_state,

            covariance=
                covariance,

            dt=
                dt,

            accelerometer_measurement_eci=
                accelerometer_measurements[
                    index - 1
                ],

            accelerometer_noise_std=
                accelerometer_noise_std
        )


        if gnss_available[
            index
        ]:

            update_result = (
                ekf_update_position(
                    predicted_state=
                        predicted_state,

                    predicted_covariance=
                        predicted_covariance,

                    position_measurement=
                        gnss_measurements[
                            index
                        ],

                    measurement_noise_covariance=
                        R
                )
            )


            innovation = (
                update_result[
                    "innovation"
                ]
            )


            innovation_covariance = (
                update_result[
                    "innovation_covariance"
                ]
            )


            nis_values[
                index
            ] = (
                innovation.T
                @ np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )


            estimated_state = (
                update_result[
                    "state"
                ]
            )


            covariance = (
                update_result[
                    "covariance"
                ]
            )


        else:

            estimated_state = (
                predicted_state
            )

            covariance = (
                predicted_covariance
            )


        estimation_error = (
            estimated_state
            - truth_states[
                index
            ]
        )


        position_errors[
            index
        ] = np.linalg.norm(
            estimation_error[
                0:3
            ]
        )


        position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        nees_values[
            index
        ] = (
            estimation_error.T
            @ np.linalg.solve(
                covariance,
                estimation_error
            )
        )


    return {
        "name":
            "6D - biais ignore",

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "nees":
            nees_values,

        "nis":
            nis_values
    }


# ------------------------------------------------------------
# 14. FILTRE 9D :
# ESTIMATION DU BIAIS
# ------------------------------------------------------------

def run_filter_with_bias_estimation():

    estimated_state = (
        estimated_initial_augmented_state.copy()
    )


    covariance = (
        P0_9.copy()
    )


    position_errors = np.zeros(
        number_of_epochs
    )


    position_sigmas = np.zeros(
        number_of_epochs
    )


    bias_estimates = np.zeros(
        (
            number_of_epochs,
            3
        )
    )


    bias_error_norms = np.zeros(
        number_of_epochs
    )


    nees_values = np.full(
        number_of_epochs,
        np.nan
    )


    nis_values = np.full(
        number_of_epochs,
        np.nan
    )


    true_initial_augmented_state = (
        np.concatenate(
            (
                truth_states[0],
                true_accelerometer_bias
            )
        )
    )


    initial_error = (
        estimated_state
        - true_initial_augmented_state
    )


    position_errors[0] = np.linalg.norm(
        initial_error[
            0:3
        ]
    )


    position_sigmas[0] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    bias_estimates[0] = (
        estimated_state[
            6:9
        ]
    )


    bias_error_norms[0] = np.linalg.norm(
        estimated_state[
            6:9
        ]
        - true_accelerometer_bias
    )


    nees_values[0] = (
        initial_error.T
        @ np.linalg.solve(
            covariance,
            initial_error
        )
    )


    for index in range(
        1,
        number_of_epochs
    ):

        (
            predicted_state,
            predicted_covariance
        ) = ekf_predict_with_bias_estimation(
            state=
                estimated_state,

            covariance=
                covariance,

            dt=
                dt,

            accelerometer_measurement_eci=
                accelerometer_measurements[
                    index - 1
                ],

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                bias_random_walk_density
        )


        if gnss_available[
            index
        ]:

            update_result = (
                ekf_update_position_augmented(
                    predicted_state=
                        predicted_state,

                    predicted_covariance=
                        predicted_covariance,

                    position_measurement=
                        gnss_measurements[
                            index
                        ],

                    measurement_noise_covariance=
                        R
                )
            )


            innovation = (
                update_result[
                    "innovation"
                ]
            )


            innovation_covariance = (
                update_result[
                    "innovation_covariance"
                ]
            )


            nis_values[
                index
            ] = (
                innovation.T
                @ np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )


            estimated_state = (
                update_result[
                    "state"
                ]
            )


            covariance = (
                update_result[
                    "covariance"
                ]
            )


        else:

            estimated_state = (
                predicted_state
            )

            covariance = (
                predicted_covariance
            )


        true_augmented_state = (
            np.concatenate(
                (
                    truth_states[
                        index
                    ],
                    true_accelerometer_bias
                )
            )
        )


        estimation_error = (
            estimated_state
            - true_augmented_state
        )


        position_errors[
            index
        ] = np.linalg.norm(
            estimation_error[
                0:3
            ]
        )


        position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        bias_estimates[
            index
        ] = (
            estimated_state[
                6:9
            ]
        )


        bias_error_norms[
            index
        ] = np.linalg.norm(
            estimated_state[
                6:9
            ]
            - true_accelerometer_bias
        )


        nees_values[
            index
        ] = (
            estimation_error.T
            @ np.linalg.solve(
                covariance,
                estimation_error
            )
        )


    return {
        "name":
            "9D - biais estime",

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "bias_estimates":
            bias_estimates,

        "bias_error_norms":
            bias_error_norms,

        "nees":
            nees_values,

        "nis":
            nis_values
    }


# ------------------------------------------------------------
# 15. EXECUTION
# ------------------------------------------------------------

result_6d = (
    run_filter_without_bias_estimation()
)


result_9d = (
    run_filter_with_bias_estimation()
)


# ------------------------------------------------------------
# 16. FONCTION DE STATISTIQUES
# ------------------------------------------------------------

def compute_statistics(
    result,
    expected_nees_dimension
):

    evaluation_mask = np.ones(
        number_of_epochs,
        dtype=bool
    )


    evaluation_mask[0] = False


    available_mask = (
        gnss_available
        & evaluation_mask
    )


    rmse_available = np.sqrt(
        np.mean(
            result[
                "position_errors"
            ][
                available_mask
            ]**2
        )
    )


    rmse_outage = np.sqrt(
        np.mean(
            result[
                "position_errors"
            ][
                outage_mask
            ]**2
        )
    )


    outage_indices = np.where(
        outage_mask
    )[0]


    outage_end_index = (
        outage_indices[-1]
    )


    error_end_outage = (
        result[
            "position_errors"
        ][
            outage_end_index
        ]
    )


    sigma_end_outage = (
        result[
            "position_sigmas"
        ][
            outage_end_index
        ]
    )


    valid_nis = np.isfinite(
        result[
            "nis"
        ]
    )


    valid_nees = np.isfinite(
        result[
            "nees"
        ]
    )


    mean_nis = np.mean(
        result[
            "nis"
        ][
            valid_nis
        ]
    )


    mean_nees = np.mean(
        result[
            "nees"
        ][
            valid_nees
        ]
    )


    return_indices = np.where(
        time_minutes
        > gnss_outage_end_minutes
    )[0]


    return_error = (
        result[
            "position_errors"
        ][
            return_indices[0]
        ]
    )


    return {
        "rmse_available":
            rmse_available,

        "rmse_outage":
            rmse_outage,

        "error_end_outage":
            error_end_outage,

        "sigma_end_outage":
            sigma_end_outage,

        "mean_nis":
            mean_nis,

        "mean_nees":
            mean_nees,

        "expected_nees":
            expected_nees_dimension,

        "return_error":
            return_error
    }


stats_6d = compute_statistics(
    result=
        result_6d,

    expected_nees_dimension=
        6
)


stats_9d = compute_statistics(
    result=
        result_9d,

    expected_nees_dimension=
        9
)


# ------------------------------------------------------------
# 17. BIAIS A DES INSTANTS IMPORTANTS
# ------------------------------------------------------------

pre_outage_indices = np.where(
    time_minutes
    < gnss_outage_start_minutes
)[0]


pre_outage_index = (
    pre_outage_indices[-1]
)


outage_indices = np.where(
    outage_mask
)[0]


outage_end_index = (
    outage_indices[-1]
)


bias_before_outage = (
    result_9d[
        "bias_estimates"
    ][
        pre_outage_index
    ]
)


bias_end_outage = (
    result_9d[
        "bias_estimates"
    ][
        outage_end_index
    ]
)


bias_final = (
    result_9d[
        "bias_estimates"
    ][
        -1
    ]
)


bias_error_before_outage = np.linalg.norm(
    bias_before_outage
    - true_accelerometer_bias
)


bias_error_end_outage = np.linalg.norm(
    bias_end_outage
    - true_accelerometer_bias
)


bias_error_final = np.linalg.norm(
    bias_final
    - true_accelerometer_bias
)


# ------------------------------------------------------------
# 18. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)


print(
    "AURORA — Expérience 008-B"
)


print(
    "Estimation du biais accéléromètre par EKF augmenté"
)


print(
    "============================================================================================"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print(
    f"Biais vrai : "
    f"[{true_accelerometer_bias[0]:.3e}, "
    f"{true_accelerometer_bias[1]:.3e}, "
    f"{true_accelerometer_bias[2]:.3e}] m/s^2"
)


print(
    f"Norme biais vrai : "
    f"{np.linalg.norm(true_accelerometer_bias):.6e} m/s^2"
)


print(
    f"Sigma initial biais EKF : "
    f"{initial_bias_sigma:.6e} m/s^2"
)


print(
    f"Random walk biais : "
    f"{bias_random_walk_density:.6e} m/s^2/sqrt(s)"
)


print()


print(
    "Filtre              | RMSE GNSS | RMSE outage | "
    "Err fin | Sigma fin | NIS moy | NEES moy | Attendu | Retour"
)


print(
    "                    |    [m]    |     [m]     | "
    "  [m]    |   [m]     |         |          |         |  [m]"
)


print(
    "--------------------------------------------------------------------------------------------"
)


print(
    f"{result_6d['name']:<19} | "
    f"{stats_6d['rmse_available']:>9.3f} | "
    f"{stats_6d['rmse_outage']:>11.3f} | "
    f"{stats_6d['error_end_outage']:>7.3f} | "
    f"{stats_6d['sigma_end_outage']:>9.3f} | "
    f"{stats_6d['mean_nis']:>7.3f} | "
    f"{stats_6d['mean_nees']:>8.3f} | "
    f"{stats_6d['expected_nees']:>7d} | "
    f"{stats_6d['return_error']:>7.3f}"
)


print(
    f"{result_9d['name']:<19} | "
    f"{stats_9d['rmse_available']:>9.3f} | "
    f"{stats_9d['rmse_outage']:>11.3f} | "
    f"{stats_9d['error_end_outage']:>7.3f} | "
    f"{stats_9d['sigma_end_outage']:>9.3f} | "
    f"{stats_9d['mean_nis']:>7.3f} | "
    f"{stats_9d['mean_nees']:>8.3f} | "
    f"{stats_9d['expected_nees']:>7d} | "
    f"{stats_9d['return_error']:>7.3f}"
)


print(
    "============================================================================================"
)


print()


print(
    "----- ESTIMATION DU BIAIS -----"
)


print(
    "Avant coupure : "
    f"[{bias_before_outage[0]:.3e}, "
    f"{bias_before_outage[1]:.3e}, "
    f"{bias_before_outage[2]:.3e}] m/s^2"
)


print(
    f"Erreur norme avant coupure : "
    f"{bias_error_before_outage:.6e} m/s^2"
)


print()


print(
    "Fin coupure : "
    f"[{bias_end_outage[0]:.3e}, "
    f"{bias_end_outage[1]:.3e}, "
    f"{bias_end_outage[2]:.3e}] m/s^2"
)


print(
    f"Erreur norme fin coupure : "
    f"{bias_error_end_outage:.6e} m/s^2"
)


print()


print(
    "Fin simulation : "
    f"[{bias_final[0]:.3e}, "
    f"{bias_final[1]:.3e}, "
    f"{bias_final[2]:.3e}] m/s^2"
)


print(
    f"Erreur norme finale : "
    f"{bias_error_final:.6e} m/s^2"
)


# ------------------------------------------------------------
# 19. FIGURE ERREUR POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    result_6d[
        "position_errors"
    ],
    label="6D - biais ignoré"
)


plt.plot(
    time_minutes,
    result_9d[
        "position_errors"
    ],
    label="9D - biais estimé"
)


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15,
    label="Coupure GNSS"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur position 3D [m]"
)


plt.title(
    "AURORA — Effet de l'estimation du biais accéléromètre"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. FIGURE ESTIMATION DU BIAIS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


labels = [
    "b_ax",
    "b_ay",
    "b_az"
]


for axis in range(
    3
):

    plt.plot(
        time_minutes,
        result_9d[
            "bias_estimates"
        ][
            :,
            axis
        ],
        label=
            f"{labels[axis]} estimé"
    )


    plt.axhline(
        true_accelerometer_bias[
            axis
        ],
        linestyle="--"
    )


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Biais [m/s²]"
)


plt.title(
    "AURORA — Convergence du biais accéléromètre"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. FIGURE ERREUR DE BIAIS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    result_9d[
        "bias_error_norms"
    ]
)


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Norme erreur biais [m/s²]"
)


plt.title(
    "AURORA — Erreur d'estimation du biais accéléromètre"
)


plt.grid(True)

plt.tight_layout()

plt.show()