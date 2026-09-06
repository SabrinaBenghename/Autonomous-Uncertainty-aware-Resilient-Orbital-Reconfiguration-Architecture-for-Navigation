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
# Expérience 008-C
#
# Convergence du biais accéléromètre
# AVANT la coupure GNSS
#
# Comparaison :
#
# 6D : biais ignoré
# 9D : biais estimé
#
# GNSS disponible 60 min avant l'outage.
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

simulation_duration_minutes = (
    100.0
)

simulation_duration = (
    simulation_duration_minutes
    * 60.0
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
    60.0
)

gnss_outage_end_minutes = (
    80.0
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
# 7. GNSS
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
                time[index],

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
# 9. ETAT INITIAL ESTIME
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
# 10. COVARIANCE 6D
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
# 11. INITIALISATION 9D
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
# 12. RANDOM WALK DU BIAIS
# ------------------------------------------------------------

bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 13. FILTRE 6D
# ------------------------------------------------------------

def run_filter_6d():

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


    return {
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas
    }


# ------------------------------------------------------------
# 14. FILTRE 9D
# ------------------------------------------------------------

def run_filter_9d():

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


    bias_sigma_3d = np.zeros(
        number_of_epochs
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


    bias_sigma_3d[0] = np.sqrt(
        np.trace(
            covariance[
                6:9,
                6:9
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


        bias_sigma_3d[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    6:9,
                    6:9
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
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "bias_estimates":
            bias_estimates,

        "bias_error_norms":
            bias_error_norms,

        "bias_sigmas":
            bias_sigma_3d,

        "nees":
            nees_values,

        "nis":
            nis_values
    }


# ------------------------------------------------------------
# 15. EXECUTION
# ------------------------------------------------------------

result_6d = (
    run_filter_6d()
)


result_9d = (
    run_filter_9d()
)


# ------------------------------------------------------------
# 16. EPOQUES IMPORTANTES
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]


outage_start_index = (
    outage_indices[0]
)


outage_end_index = (
    outage_indices[-1]
)


pre_outage_index = (
    outage_start_index
    - 1
)


after_outage_indices = np.where(
    time_minutes
    > gnss_outage_end_minutes
)[0]


first_return_index = (
    after_outage_indices[0]
)


# ------------------------------------------------------------
# 17. STATISTIQUES POSITION
# ------------------------------------------------------------

evaluation_mask = np.ones(
    number_of_epochs,
    dtype=bool
)

evaluation_mask[0] = False


available_mask = (
    gnss_available
    & evaluation_mask
)


rmse_6d_available = np.sqrt(
    np.mean(
        result_6d[
            "position_errors"
        ][
            available_mask
        ]**2
    )
)


rmse_9d_available = np.sqrt(
    np.mean(
        result_9d[
            "position_errors"
        ][
            available_mask
        ]**2
    )
)


rmse_6d_outage = np.sqrt(
    np.mean(
        result_6d[
            "position_errors"
        ][
            outage_mask
        ]**2
    )
)


rmse_9d_outage = np.sqrt(
    np.mean(
        result_9d[
            "position_errors"
        ][
            outage_mask
        ]**2
    )
)


# ------------------------------------------------------------
# 18. BIAIS AU DEBUT DE LA COUPURE
# ------------------------------------------------------------

estimated_bias_before_outage = (
    result_9d[
        "bias_estimates"
    ][
        pre_outage_index
    ]
)


bias_error_before_outage = (
    result_9d[
        "bias_error_norms"
    ][
        pre_outage_index
    ]
)


bias_sigma_before_outage = (
    result_9d[
        "bias_sigmas"
    ][
        pre_outage_index
    ]
)


estimated_bias_end_outage = (
    result_9d[
        "bias_estimates"
    ][
        outage_end_index
    ]
)


bias_error_end_outage = (
    result_9d[
        "bias_error_norms"
    ][
        outage_end_index
    ]
)


# ------------------------------------------------------------
# 19. NIS / NEES 9D
# ------------------------------------------------------------

valid_nis = np.isfinite(
    result_9d[
        "nis"
    ]
)


valid_nees = np.isfinite(
    result_9d[
        "nees"
    ]
)


mean_nis = np.mean(
    result_9d[
        "nis"
    ][
        valid_nis
    ]
)


mean_nees = np.mean(
    result_9d[
        "nees"
    ][
        valid_nees
    ]
)


# ------------------------------------------------------------
# 20. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)


print(
    "AURORA — Expérience 008-C"
)


print(
    "Convergence du biais avant coupure GNSS"
)


print(
    "============================================================================================"
)


print(
    f"Durée totale : "
    f"{simulation_duration_minutes:.1f} min"
)


print(
    f"Coupure GNSS : "
    f"{gnss_outage_start_minutes:.1f} "
    f"à {gnss_outage_end_minutes:.1f} min"
)


print(
    f"Temps d'apprentissage du biais avant coupure : "
    f"{gnss_outage_start_minutes:.1f} min"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- BIAIS VRAI -----"
)


print(
    f"[{true_accelerometer_bias[0]:.3e}, "
    f"{true_accelerometer_bias[1]:.3e}, "
    f"{true_accelerometer_bias[2]:.3e}] m/s^2"
)


print(
    f"Norme : "
    f"{np.linalg.norm(true_accelerometer_bias):.6e} m/s^2"
)


print()


print(
    "----- BIAIS ESTIME AVANT COUPURE -----"
)


print(
    f"[{estimated_bias_before_outage[0]:.3e}, "
    f"{estimated_bias_before_outage[1]:.3e}, "
    f"{estimated_bias_before_outage[2]:.3e}] m/s^2"
)


print(
    f"Erreur norme : "
    f"{bias_error_before_outage:.6e} m/s^2"
)


print(
    f"Incertitude biais 3D : "
    f"{bias_sigma_before_outage:.6e} m/s^2"
)


print()


print(
    "----- PERFORMANCE POSITION -----"
)


print(
    "Filtre          | RMSE GNSS | RMSE outage | "
    "Err fin outage | Retour GNSS"
)


print(
    "                |    [m]    |     [m]     | "
    "     [m]       |    [m]"
)


print(
    "------------------------------------------------------------------------"
)


print(
    f"{'6D biais ignore':<15} | "
    f"{rmse_6d_available:>9.3f} | "
    f"{rmse_6d_outage:>11.3f} | "
    f"{result_6d['position_errors'][outage_end_index]:>13.3f} | "
    f"{result_6d['position_errors'][first_return_index]:>10.3f}"
)


print(
    f"{'9D biais estime':<15} | "
    f"{rmse_9d_available:>9.3f} | "
    f"{rmse_9d_outage:>11.3f} | "
    f"{result_9d['position_errors'][outage_end_index]:>13.3f} | "
    f"{result_9d['position_errors'][first_return_index]:>10.3f}"
)


print()


print(
    "----- COHERENCE 9D -----"
)


print(
    f"NIS moyen : "
    f"{mean_nis:.3f} "
    f"(attendu ~3)"
)


print(
    f"NEES moyen : "
    f"{mean_nees:.3f} "
    f"(attendu ~9)"
)


print()


print(
    "----- BIAIS FIN COUPURE -----"
)


print(
    f"[{estimated_bias_end_outage[0]:.3e}, "
    f"{estimated_bias_end_outage[1]:.3e}, "
    f"{estimated_bias_end_outage[2]:.3e}] m/s^2"
)


print(
    f"Erreur norme : "
    f"{bias_error_end_outage:.6e} m/s^2"
)


print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 21. FIGURE POSITION
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
    "AURORA — Navigation après convergence du biais"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. FIGURE ESTIMATION DES 3 BIAIS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


bias_labels = [
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
            f"{bias_labels[axis]} estimé"
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
# 23. FIGURE ERREUR + INCERTITUDE DU BIAIS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    result_9d[
        "bias_error_norms"
    ],
    label="Erreur réelle du biais"
)


plt.plot(
    time_minutes,
    result_9d[
        "bias_sigmas"
    ],
    linestyle="--",
    label="Incertitude biais 3D"
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
    "AURORA — Erreur et incertitude du biais"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()