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

from src.navigation.orbit_ekf_models import (
    j2_state_derivative,
    ekf_predict_variational,
    build_discrete_acceleration_process_noise
)

from src.navigation.imu_aided_ekf import (
    ekf_predict_with_accelerometer
)

from src.sensors.accelerometer import (
    simulate_accelerometer_measurement
)


# ============================================================
# AURORA
# Expérience 008-A
#
# Navigation GNSS + accéléromètre
#
# Vérité :
#   gravité + J2 + perturbation non gravitationnelle
#
# Comparaison :
#
# A - J2 seul, Q = 0
# B - J2 + bruit de processus
# C - J2 + accéléromètre
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH + 550_000.0
)

eccentricity = 0.01

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
# 4. PERTURBATION VRAIE
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

accelerometer_bias = np.array([
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

    result = (
        simulate_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force,

            bias_eci=
                accelerometer_bias,

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
        result[
            "measurement"
        ]
    )


# ------------------------------------------------------------
# 9. STATISTIQUES ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_errors = (
    accelerometer_measurements
    - true_specific_forces
)

accelerometer_error_rms = np.sqrt(
    np.mean(
        np.sum(
            accelerometer_errors**2,
            axis=1
        )
    )
)

mean_true_specific_force = np.mean(
    np.linalg.norm(
        true_specific_forces,
        axis=1
    )
)


# ------------------------------------------------------------
# 10. INITIALISATION FILTRE
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

estimated_initial_state = (
    true_initial_state
    +
    np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )
)


initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)

P0 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2
])


# ------------------------------------------------------------
# 11. Q DES FILTRES SANS ACCELEROMETRE
# ------------------------------------------------------------

Q_zero = np.zeros(
    (
        6,
        6
    )
)


equivalent_process_acceleration_sigma = (
    2.0e-5
)


Q_without_accelerometer = (
    build_discrete_acceleration_process_noise(
        dt=
            dt,

        acceleration_sigma=
            equivalent_process_acceleration_sigma
    )
)


# ------------------------------------------------------------
# 12. EXECUTION D'UN FILTRE
# ------------------------------------------------------------

def run_filter(
    name,
    mode
):

    estimated_state = (
        estimated_initial_state.copy()
    )

    covariance = (
        P0.copy()
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


    # --------------------------------------------------------
    # Boucle temporelle
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if mode == "j2_q0":

            (
                predicted_state,
                predicted_covariance
            ) = ekf_predict_variational(
                state=
                    estimated_state,

                covariance=
                    covariance,

                dt=
                    dt,

                process_noise_covariance=
                    Q_zero,

                dynamics_function=
                    j2_state_derivative
            )


        elif mode == "j2_q":

            (
                predicted_state,
                predicted_covariance
            ) = ekf_predict_variational(
                state=
                    estimated_state,

                covariance=
                    covariance,

                dt=
                    dt,

                process_noise_covariance=
                    Q_without_accelerometer,

                dynamics_function=
                    j2_state_derivative
            )


        elif mode == "accelerometer":

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


        else:

            raise ValueError(
                "Mode de filtre inconnu."
            )


        # ----------------------------------------------------
        # CORRECTION GNSS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ERREUR
        # ----------------------------------------------------

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

        try:

            nees_values[
                index
            ] = (
                estimation_error.T
                @ np.linalg.solve(
                    covariance,
                    estimation_error
                )
            )

        except np.linalg.LinAlgError:

            nees_values[
                index
            ] = np.nan


    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------

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
            position_errors[
                available_mask
            ]**2
        )
    )


    rmse_outage = np.sqrt(
        np.mean(
            position_errors[
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
        position_errors[
            outage_end_index
        ]
    )


    sigma_end_outage = (
        position_sigmas[
            outage_end_index
        ]
    )


    valid_nis = np.isfinite(
        nis_values
    )

    valid_nees = np.isfinite(
        nees_values
    )


    mean_nis = np.mean(
        nis_values[
            valid_nis
        ]
    )


    mean_nees = np.mean(
        nees_values[
            valid_nees
        ]
    )


    return_indices = np.where(
        time_minutes
        > gnss_outage_end_minutes
    )[0]


    return_error = (
        position_errors[
            return_indices[0]
        ]
    )


    return {
        "name":
            name,

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "nees":
            nees_values,

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

        "return_error":
            return_error
    }


# ------------------------------------------------------------
# 13. TROIS CONFIGURATIONS
# ------------------------------------------------------------

result_j2_q0 = run_filter(
    name=
        "J2, Q=0",

    mode=
        "j2_q0"
)


result_j2_q = run_filter(
    name=
        "J2 + Q",

    mode=
        "j2_q"
)


result_accelerometer = run_filter(
    name=
        "J2 + accelerometer",

    mode=
        "accelerometer"
)


results = [
    result_j2_q0,
    result_j2_q,
    result_accelerometer
]


# ------------------------------------------------------------
# 14. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=========================================================================================="
)

print(
    "AURORA — Expérience 008-A"
)

print(
    "Navigation GNSS + accéléromètre"
)

print(
    "=========================================================================================="
)


print(
    f"Perturbation non gravitationnelle moyenne : "
    f"{mean_true_specific_force:.6e} m/s^2"
)


print(
    f"Biais accéléromètre : "
    f"{np.linalg.norm(accelerometer_bias):.6e} m/s^2"
)


print(
    f"Bruit accéléromètre sigma : "
    f"{accelerometer_noise_std:.6e} m/s^2"
)


print(
    f"RMS erreur mesure accéléromètre 3D : "
    f"{accelerometer_error_rms:.6e} m/s^2"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "Modèle               | RMSE GNSS | RMSE outage | "
    "Err fin | Sigma fin | NIS moy. | NEES moy. | Retour"
)


print(
    "                     |    [m]    |     [m]     | "
    "  [m]    |   [m]     |          |           |  [m]"
)


print(
    "------------------------------------------------------------------------------------------"
)


for result in results:

    print(
        f"{result['name']:<20} | "
        f"{result['rmse_available']:>9.3f} | "
        f"{result['rmse_outage']:>11.3f} | "
        f"{result['error_end_outage']:>7.3f} | "
        f"{result['sigma_end_outage']:>9.3f} | "
        f"{result['mean_nis']:>8.3f} | "
        f"{result['mean_nees']:>9.3f} | "
        f"{result['return_error']:>7.3f}"
    )


print(
    "=========================================================================================="
)


# ------------------------------------------------------------
# 15. FIGURE ERREUR POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


for result in results:

    plt.plot(
        time_minutes,
        result[
            "position_errors"
        ],
        label=
            result[
                "name"
            ]
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
    "AURORA — Bénéfice de l'accéléromètre pendant une coupure GNSS"
)


plt.yscale(
    "log"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FIGURE INCERTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


for result in results:

    plt.plot(
        time_minutes,
        result[
            "position_sigmas"
        ],
        label=
            result[
                "name"
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
    "Incertitude position 3D [m]"
)


plt.title(
    "AURORA — Incertitude de navigation"
)


plt.yscale(
    "log"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. FIGURE FORCE SPECIFIQUE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    np.linalg.norm(
        true_specific_forces,
        axis=1
    ),
    label="Force spécifique vraie"
)


plt.plot(
    time_minutes,
    np.linalg.norm(
        accelerometer_measurements,
        axis=1
    ),
    label="Mesure accéléromètre"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Accélération [m/s²]"
)


plt.title(
    "AURORA — Accélération non gravitationnelle mesurée"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()