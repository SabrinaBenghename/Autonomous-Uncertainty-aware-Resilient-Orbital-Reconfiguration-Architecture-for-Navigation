import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

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

from src.navigation.accelerometer_bias_ekf import (
    ekf_predict_with_bias_estimation,
    ekf_update_position_augmented
)

from src.sensors.accelerometer import (
    simulate_accelerometer_measurement
)


# ============================================================
# AURORA
# Expérience 008-E
#
# Diagnostic de cohérence par sous-état :
#
# - position
# - vitesse
# - biais accéléromètre
#
# Objectif :
# déterminer quelle composante explique
# le léger excès de NEES observé en 008-D.
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH
    + 550_000.0
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
    time
    / 60.0
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
# 4. MONTE CARLO
# ------------------------------------------------------------

number_of_runs = (
    50
)

master_rng = np.random.default_rng(
    2026
)

run_seeds = (
    master_rng.integers(
        low=0,
        high=2**32 - 1,
        size=number_of_runs
    )
)


# ------------------------------------------------------------
# 5. PERTURBATION NON GRAVITATIONNELLE
# ------------------------------------------------------------

true_non_gravitational_acceleration = (
    2.0e-5
)


# ------------------------------------------------------------
# 6. ETAT INITIAL VRAI
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

true_initial_orbital_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)


# ------------------------------------------------------------
# 7. VERITE ORBITALE
# ------------------------------------------------------------

truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_orbital_state,

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
# 8. FORCE SPECIFIQUE VRAIE
# ------------------------------------------------------------

true_specific_forces = np.zeros(
    (
        number_of_epochs,
        3
    )
)

for index in range(
    number_of_epochs
):

    true_specific_forces[
        index
    ] = (
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


# ------------------------------------------------------------
# 9. BRUITS / COVARIANCES
# ------------------------------------------------------------

gnss_position_sigma = (
    3.0
)

R = (
    gnss_position_sigma**2
    * np.eye(3)
)

accelerometer_noise_std = (
    5.0e-7
)

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)

initial_bias_sigma = (
    5.0e-6
)

bias_random_walk_density = (
    1.0e-10
)

P0 = np.diag([
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
# 10. STOCKAGE
# ------------------------------------------------------------

full_nees_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

position_nees_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

velocity_nees_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

bias_nees_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


position_error_norms = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

velocity_error_norms = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

bias_error_norms = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


position_sigma_3d = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

velocity_sigma_3d = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)

bias_sigma_3d = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


# ------------------------------------------------------------
# 11. FONCTION DE CALCUL DU NEES
# ------------------------------------------------------------

def compute_nees(
    error,
    covariance
):

    try:

        return (
            error.T
            @ np.linalg.solve(
                covariance,
                error
            )
        )

    except np.linalg.LinAlgError:

        return np.nan


# ------------------------------------------------------------
# 12. BOUCLE MONTE CARLO
# ------------------------------------------------------------

for run_index in range(
    number_of_runs
):

    rng = np.random.default_rng(
        run_seeds[
            run_index
        ]
    )


    # --------------------------------------------------------
    # Biais vrai compatible avec le prior
    # --------------------------------------------------------

    true_bias = rng.normal(
        loc=0.0,
        scale=initial_bias_sigma,
        size=3
    )


    # --------------------------------------------------------
    # Erreur initiale orbitale
    # --------------------------------------------------------

    initial_position_error = rng.normal(
        loc=0.0,
        scale=initial_position_sigma,
        size=3
    )

    initial_velocity_error = rng.normal(
        loc=0.0,
        scale=initial_velocity_sigma,
        size=3
    )


    estimated_state = np.concatenate(
        (
            true_initial_orbital_state
            + np.concatenate(
                (
                    initial_position_error,
                    initial_velocity_error
                )
            ),

            np.zeros(3)
        )
    )


    covariance = (
        P0.copy()
    )


    # --------------------------------------------------------
    # GNSS
    # --------------------------------------------------------

    gnss_measurements = (
        truth_states[:, 0:3]
        +
        rng.normal(
            loc=0.0,
            scale=gnss_position_sigma,
            size=(
                number_of_epochs,
                3
            )
        )
    )


    # --------------------------------------------------------
    # ACCELEROMETRE
    # --------------------------------------------------------

    accelerometer_measurements = np.zeros(
        (
            number_of_epochs,
            3
        )
    )


    for index in range(
        number_of_epochs
    ):

        measurement_result = (
            simulate_accelerometer_measurement(
                true_specific_force_eci=
                    true_specific_forces[
                        index
                    ],

                bias_eci=
                    true_bias,

                noise_std=
                    accelerometer_noise_std,

                rng=
                    rng
            )
        )


        accelerometer_measurements[
            index
        ] = (
            measurement_result[
                "measurement"
            ]
        )


    # --------------------------------------------------------
    # BOUCLE TEMPORELLE
    # --------------------------------------------------------

    for index in range(
        number_of_epochs
    ):

        if index > 0:

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
        # VRAI ETAT AUGMENTE
        # ----------------------------------------------------

        true_augmented_state = np.concatenate(
            (
                truth_states[
                    index
                ],

                true_bias
            )
        )


        error = (
            estimated_state
            - true_augmented_state
        )


        position_error = (
            error[
                0:3
            ]
        )

        velocity_error = (
            error[
                3:6
            ]
        )

        bias_error = (
            error[
                6:9
            ]
        )


        P_position = (
            covariance[
                0:3,
                0:3
            ]
        )

        P_velocity = (
            covariance[
                3:6,
                3:6
            ]
        )

        P_bias = (
            covariance[
                6:9,
                6:9
            ]
        )


        # ----------------------------------------------------
        # NEES COMPLET
        # ----------------------------------------------------

        full_nees_matrix[
            run_index,
            index
        ] = compute_nees(
            error=
                error,

            covariance=
                covariance
        )


        # ----------------------------------------------------
        # NEES POSITION
        # ----------------------------------------------------

        position_nees_matrix[
            run_index,
            index
        ] = compute_nees(
            error=
                position_error,

            covariance=
                P_position
        )


        # ----------------------------------------------------
        # NEES VITESSE
        # ----------------------------------------------------

        velocity_nees_matrix[
            run_index,
            index
        ] = compute_nees(
            error=
                velocity_error,

            covariance=
                P_velocity
        )


        # ----------------------------------------------------
        # NEES BIAIS
        # ----------------------------------------------------

        bias_nees_matrix[
            run_index,
            index
        ] = compute_nees(
            error=
                bias_error,

            covariance=
                P_bias
        )


        # ----------------------------------------------------
        # ERREURS
        # ----------------------------------------------------

        position_error_norms[
            run_index,
            index
        ] = np.linalg.norm(
            position_error
        )


        velocity_error_norms[
            run_index,
            index
        ] = np.linalg.norm(
            velocity_error
        )


        bias_error_norms[
            run_index,
            index
        ] = np.linalg.norm(
            bias_error
        )


        # ----------------------------------------------------
        # INCERTITUDES 3D
        # ----------------------------------------------------

        position_sigma_3d[
            run_index,
            index
        ] = np.sqrt(
            np.trace(
                P_position
            )
        )


        velocity_sigma_3d[
            run_index,
            index
        ] = np.sqrt(
            np.trace(
                P_velocity
            )
        )


        bias_sigma_3d[
            run_index,
            index
        ] = np.sqrt(
            np.trace(
                P_bias
            )
        )


# ------------------------------------------------------------
# 13. ANEES PAR EPOQUE
# ------------------------------------------------------------

full_anees = np.nanmean(
    full_nees_matrix,
    axis=0
)

position_anees = np.nanmean(
    position_nees_matrix,
    axis=0
)

velocity_anees = np.nanmean(
    velocity_nees_matrix,
    axis=0
)

bias_anees = np.nanmean(
    bias_nees_matrix,
    axis=0
)


# ------------------------------------------------------------
# 14. MOYENNES GLOBALES
# ------------------------------------------------------------

mean_full_nees = np.nanmean(
    full_nees_matrix
)

mean_position_nees = np.nanmean(
    position_nees_matrix
)

mean_velocity_nees = np.nanmean(
    velocity_nees_matrix
)

mean_bias_nees = np.nanmean(
    bias_nees_matrix
)


mean_full_nees_outage = np.nanmean(
    full_nees_matrix[
        :,
        outage_mask
    ]
)

mean_position_nees_outage = np.nanmean(
    position_nees_matrix[
        :,
        outage_mask
    ]
)

mean_velocity_nees_outage = np.nanmean(
    velocity_nees_matrix[
        :,
        outage_mask
    ]
)

mean_bias_nees_outage = np.nanmean(
    bias_nees_matrix[
        :,
        outage_mask
    ]
)


# ------------------------------------------------------------
# 15. BORNES ANEES
# ------------------------------------------------------------

alpha = (
    0.05
)


def ensemble_bounds(
    dimension
):

    degrees_of_freedom = (
        number_of_runs
        * dimension
    )


    lower = (
        chi2.ppf(
            alpha / 2.0,
            degrees_of_freedom
        )
        / number_of_runs
    )


    upper = (
        chi2.ppf(
            1.0 - alpha / 2.0,
            degrees_of_freedom
        )
        / number_of_runs
    )


    return (
        lower,
        upper
    )


full_lower, full_upper = (
    ensemble_bounds(
        9
    )
)

component_lower, component_upper = (
    ensemble_bounds(
        3
    )
)


# ------------------------------------------------------------
# 16. COUVERTURE ANEES
# ------------------------------------------------------------

def compute_coverage(
    values,
    lower,
    upper
):

    return (
        100.0
        * np.mean(
            (
                values >= lower
            )
            &
            (
                values <= upper
            )
        )
    )


full_coverage = (
    compute_coverage(
        full_anees,
        full_lower,
        full_upper
    )
)

position_coverage = (
    compute_coverage(
        position_anees,
        component_lower,
        component_upper
    )
)

velocity_coverage = (
    compute_coverage(
        velocity_anees,
        component_lower,
        component_upper
    )
)

bias_coverage = (
    compute_coverage(
        bias_anees,
        component_lower,
        component_upper
    )
)


# ------------------------------------------------------------
# 17. RMSE / SIGMA PAR COMPOSANTE
# ------------------------------------------------------------

position_rmse_time = np.sqrt(
    np.mean(
        position_error_norms**2,
        axis=0
    )
)

velocity_rmse_time = np.sqrt(
    np.mean(
        velocity_error_norms**2,
        axis=0
    )
)

bias_rmse_time = np.sqrt(
    np.mean(
        bias_error_norms**2,
        axis=0
    )
)


mean_position_sigma_time = np.mean(
    position_sigma_3d,
    axis=0
)

mean_velocity_sigma_time = np.mean(
    velocity_sigma_3d,
    axis=0
)

mean_bias_sigma_time = np.mean(
    bias_sigma_3d,
    axis=0
)


# ------------------------------------------------------------
# 18. FIN DE COUPURE
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]

outage_end_index = (
    outage_indices[-1]
)


# ------------------------------------------------------------
# 19. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)

print(
    "AURORA — Expérience 008-E"
)

print(
    "Diagnostic de cohérence par composante de l'EKF 9D"
)

print(
    "============================================================================================"
)

print(
    f"Monte Carlo : "
    f"{number_of_runs} runs"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()

print(
    "----- NEES / ANEES -----"
)

print(
    "Sous-état     | Dimension | NEES moyen | "
    "NEES outage | ANEES 95% couverture"
)

print(
    "--------------------------------------------------------------------------"
)

print(
    f"{'Etat complet':<13} | "
    f"{9:>9d} | "
    f"{mean_full_nees:>10.3f} | "
    f"{mean_full_nees_outage:>11.3f} | "
    f"{full_coverage:>19.2f} %"
)

print(
    f"{'Position':<13} | "
    f"{3:>9d} | "
    f"{mean_position_nees:>10.3f} | "
    f"{mean_position_nees_outage:>11.3f} | "
    f"{position_coverage:>19.2f} %"
)

print(
    f"{'Vitesse':<13} | "
    f"{3:>9d} | "
    f"{mean_velocity_nees:>10.3f} | "
    f"{mean_velocity_nees_outage:>11.3f} | "
    f"{velocity_coverage:>19.2f} %"
)

print(
    f"{'Biais accel.':<13} | "
    f"{3:>9d} | "
    f"{mean_bias_nees:>10.3f} | "
    f"{mean_bias_nees_outage:>11.3f} | "
    f"{bias_coverage:>19.2f} %"
)

print()

print(
    f"Bornes ANEES état 9D : "
    f"[{full_lower:.3f}, "
    f"{full_upper:.3f}]"
)

print(
    f"Bornes ANEES sous-état 3D : "
    f"[{component_lower:.3f}, "
    f"{component_upper:.3f}]"
)

print()

print(
    "----- FIN DE COUPURE GNSS -----"
)

print(
    f"Position : RMSE = "
    f"{position_rmse_time[outage_end_index]:.6f} m"
)

print(
    f"Position : sigma 3D = "
    f"{mean_position_sigma_time[outage_end_index]:.6f} m"
)

print()

print(
    f"Vitesse : RMSE = "
    f"{velocity_rmse_time[outage_end_index]:.6e} m/s"
)

print(
    f"Vitesse : sigma 3D = "
    f"{mean_velocity_sigma_time[outage_end_index]:.6e} m/s"
)

print()

print(
    f"Biais : RMSE = "
    f"{bias_rmse_time[outage_end_index]:.6e} m/s^2"
)

print(
    f"Biais : sigma 3D = "
    f"{mean_bias_sigma_time[outage_end_index]:.6e} m/s^2"
)

print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 20. FIGURE ANEES DES SOUS-ETATS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    position_anees,
    label="ANEES position"
)

plt.plot(
    time_minutes,
    velocity_anees,
    label="ANEES vitesse"
)

plt.plot(
    time_minutes,
    bias_anees,
    label="ANEES biais"
)

plt.axhline(
    3.0,
    linestyle="--",
    label="Valeur théorique = 3"
)

plt.axhline(
    component_lower,
    linestyle=":"
)

plt.axhline(
    component_upper,
    linestyle=":"
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
    "ANEES [-]"
)

plt.title(
    "AURORA — Cohérence par sous-état"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. FIGURE VITESSE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    velocity_rmse_time,
    label="RMSE vitesse"
)

plt.plot(
    time_minutes,
    mean_velocity_sigma_time,
    linestyle="--",
    label="Incertitude vitesse 3D"
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
    "Vitesse [m/s]"
)

plt.title(
    "AURORA — Erreur et covariance de vitesse"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()