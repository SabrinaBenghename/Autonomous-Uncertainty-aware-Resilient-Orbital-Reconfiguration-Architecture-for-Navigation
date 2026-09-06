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
# Expérience 008-D
#
# Validation Monte Carlo du filtre 9D :
#
# x = [
#       r,
#       v,
#       biais accelerometre
#     ]
#
# Validation :
#
# - NIS
# - NEES
# - ANIS
# - ANEES
# - RMSE position
# - RMSE pendant outage
# - estimation du biais
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
# 5. FORCE NON GRAVITATIONNELLE
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
#
# La trajectoire vraie est la même pour
# tous les runs.
#
# Le biais capteur change, mais le biais
# n'affecte évidemment pas la trajectoire
# physique réelle.
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
# 9. BRUIT GNSS
# ------------------------------------------------------------

gnss_position_sigma = (
    3.0
)


R = (
    gnss_position_sigma**2
    * np.eye(3)
)


# ------------------------------------------------------------
# 10. ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


# ------------------------------------------------------------
# 11. COVARIANCE INITIALE
# ------------------------------------------------------------

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)


# ------------------------------------------------------------
# IMPORTANT
#
# Pour ce Monte Carlo statistique, le biais
# vrai est lui-même tiré selon le prior
# utilisé par l'EKF.
#
# Ainsi :
#
# b_true ~ N(0, sigma_b^2 I)
#
# et l'EKF démarre avec :
#
# b_hat = 0
#
# C'est nécessaire pour que l'hypothèse
# statistique du NEES soit cohérente.
# ------------------------------------------------------------

initial_bias_sigma = (
    5.0e-6
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
# 12. RANDOM WALK DU BIAIS
# ------------------------------------------------------------

bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 13. DIMENSIONS
# ------------------------------------------------------------

state_dimension = (
    9
)

measurement_dimension = (
    3
)


# ------------------------------------------------------------
# 14. INTERVALLES CHI2 INDIVIDUELS
# ------------------------------------------------------------

confidence_level = (
    0.95
)


alpha = (
    1.0
    - confidence_level
)


nis_lower = chi2.ppf(
    alpha / 2.0,
    measurement_dimension
)


nis_upper = chi2.ppf(
    1.0 - alpha / 2.0,
    measurement_dimension
)


nees_lower = chi2.ppf(
    alpha / 2.0,
    state_dimension
)


nees_upper = chi2.ppf(
    1.0 - alpha / 2.0,
    state_dimension
)


# ------------------------------------------------------------
# 15. BORNES ENSEMBLE ANIS / ANEES
#
# Pour N runs :
#
# N * ANEES ~ Chi2(N * nx)
#
# donc :
#
# borne = Chi2 / N
# ------------------------------------------------------------

anis_lower = (
    chi2.ppf(
        alpha / 2.0,
        number_of_runs
        * measurement_dimension
    )
    / number_of_runs
)


anis_upper = (
    chi2.ppf(
        1.0 - alpha / 2.0,
        number_of_runs
        * measurement_dimension
    )
    / number_of_runs
)


anees_lower = (
    chi2.ppf(
        alpha / 2.0,
        number_of_runs
        * state_dimension
    )
    / number_of_runs
)


anees_upper = (
    chi2.ppf(
        1.0 - alpha / 2.0,
        number_of_runs
        * state_dimension
    )
    / number_of_runs
)


# ------------------------------------------------------------
# 16. STOCKAGE MONTE CARLO
# ------------------------------------------------------------

nis_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


nees_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


position_error_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


position_sigma_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


bias_error_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


bias_sigma_matrix = np.full(
    (
        number_of_runs,
        number_of_epochs
    ),
    np.nan
)


true_biases = np.zeros(
    (
        number_of_runs,
        3
    )
)


estimated_biases_before_outage = np.zeros(
    (
        number_of_runs,
        3
    )
)


# ------------------------------------------------------------
# 17. INDICES IMPORTANTS
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


# ------------------------------------------------------------
# 18. BOUCLE MONTE CARLO
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
    # Vrai biais du run
    # --------------------------------------------------------

    true_bias = rng.normal(
        loc=0.0,
        scale=initial_bias_sigma,
        size=3
    )


    true_biases[
        run_index
    ] = (
        true_bias
    )


    # --------------------------------------------------------
    # Erreur initiale orbitale
    #
    # Elle est également tirée selon P0.
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


    estimated_initial_orbital_state = (
        true_initial_orbital_state
        +
        np.concatenate(
            (
                initial_position_error,
                initial_velocity_error
            )
        )
    )


    # --------------------------------------------------------
    # Estimation initiale du biais
    # --------------------------------------------------------

    initial_bias_estimate = np.zeros(
        3
    )


    estimated_state = np.concatenate(
        (
            estimated_initial_orbital_state,
            initial_bias_estimate
        )
    )


    covariance = (
        P0.copy()
    )


    # --------------------------------------------------------
    # Mesures GNSS du run
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
    # Mesures accéléromètre du run
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
    # Etat vrai augmenté initial
    # --------------------------------------------------------

    true_augmented_state = np.concatenate(
        (
            truth_states[0],
            true_bias
        )
    )


    initial_estimation_error = (
        estimated_state
        - true_augmented_state
    )


    position_error_matrix[
        run_index,
        0
    ] = np.linalg.norm(
        initial_estimation_error[
            0:3
        ]
    )


    bias_error_matrix[
        run_index,
        0
    ] = np.linalg.norm(
        initial_estimation_error[
            6:9
        ]
    )


    position_sigma_matrix[
        run_index,
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    bias_sigma_matrix[
        run_index,
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                6:9,
                6:9
            ]
        )
    )


    nees_matrix[
        run_index,
        0
    ] = (
        initial_estimation_error.T
        @ np.linalg.solve(
            covariance,
            initial_estimation_error
        )
    )


    # --------------------------------------------------------
    # BOUCLE TEMPORELLE
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # PREDICTION GNSS + ACCELEROMETRE
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CORRECTION GNSS
        # ----------------------------------------------------

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


            nis_matrix[
                run_index,
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
        # ERREUR VRAIE
        # ----------------------------------------------------

        true_augmented_state = np.concatenate(
            (
                truth_states[
                    index
                ],
                true_bias
            )
        )


        estimation_error = (
            estimated_state
            - true_augmented_state
        )


        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        position_error_matrix[
            run_index,
            index
        ] = np.linalg.norm(
            estimation_error[
                0:3
            ]
        )


        position_sigma_matrix[
            run_index,
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        # ----------------------------------------------------
        # BIAIS
        # ----------------------------------------------------

        bias_error_matrix[
            run_index,
            index
        ] = np.linalg.norm(
            estimation_error[
                6:9
            ]
        )


        bias_sigma_matrix[
            run_index,
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    6:9,
                    6:9
                ]
            )
        )


        # ----------------------------------------------------
        # NEES
        # ----------------------------------------------------

        try:

            nees_matrix[
                run_index,
                index
            ] = (
                estimation_error.T
                @ np.linalg.solve(
                    covariance,
                    estimation_error
                )
            )

        except np.linalg.LinAlgError:

            nees_matrix[
                run_index,
                index
            ] = np.nan


        # ----------------------------------------------------
        # BIAIS JUSTE AVANT OUTAGE
        # ----------------------------------------------------

        if index == pre_outage_index:

            estimated_biases_before_outage[
                run_index
            ] = (
                estimated_state[
                    6:9
                ]
            )


# ------------------------------------------------------------
# 19. DONNEES VALIDES
# ------------------------------------------------------------

valid_nis_values = (
    nis_matrix[
        np.isfinite(
            nis_matrix
        )
    ]
)


valid_nees_values = (
    nees_matrix[
        np.isfinite(
            nees_matrix
        )
    ]
)


# ------------------------------------------------------------
# 20. NIS GLOBAL
# ------------------------------------------------------------

mean_nis = np.mean(
    valid_nis_values
)


nis_individual_coverage = (
    100.0
    * np.mean(
        (
            valid_nis_values
            >= nis_lower
        )
        &
        (
            valid_nis_values
            <= nis_upper
        )
    )
)


# ------------------------------------------------------------
# 21. NEES GLOBAL
# ------------------------------------------------------------

mean_nees = np.mean(
    valid_nees_values
)


nees_individual_coverage = (
    100.0
    * np.mean(
        (
            valid_nees_values
            >= nees_lower
        )
        &
        (
            valid_nees_values
            <= nees_upper
        )
    )
)


# ------------------------------------------------------------
# 22. ANIS PAR EPOQUE
# ------------------------------------------------------------

anis_values = np.full(
    number_of_epochs,
    np.nan
)


for index in range(
    number_of_epochs
):

    epoch_nis = (
        nis_matrix[
            :,
            index
        ]
    )


    valid_epoch_nis = (
        epoch_nis[
            np.isfinite(
                epoch_nis
            )
        ]
    )


    if len(
        valid_epoch_nis
    ) == number_of_runs:

        anis_values[
            index
        ] = np.mean(
            valid_epoch_nis
        )


valid_anis = np.isfinite(
    anis_values
)


anis_coverage = (
    100.0
    * np.mean(
        (
            anis_values[
                valid_anis
            ]
            >= anis_lower
        )
        &
        (
            anis_values[
                valid_anis
            ]
            <= anis_upper
        )
    )
)


# ------------------------------------------------------------
# 23. ANEES PAR EPOQUE
# ------------------------------------------------------------

anees_values = np.nanmean(
    nees_matrix,
    axis=0
)


anees_coverage = (
    100.0
    * np.mean(
        (
            anees_values
            >= anees_lower
        )
        &
        (
            anees_values
            <= anees_upper
        )
    )
)


# ------------------------------------------------------------
# 24. PERFORMANCE POSITION
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


position_rmse_available = np.sqrt(
    np.mean(
        position_error_matrix[
            :,
            available_mask
        ]**2
    )
)


position_rmse_outage = np.sqrt(
    np.mean(
        position_error_matrix[
            :,
            outage_mask
        ]**2
    )
)


position_rmse_end_outage = np.sqrt(
    np.mean(
        position_error_matrix[
            :,
            outage_end_index
        ]**2
    )
)


mean_position_sigma_end_outage = (
    np.mean(
        position_sigma_matrix[
            :,
            outage_end_index
        ]
    )
)


# ------------------------------------------------------------
# 25. PERFORMANCE DU BIAIS
# ------------------------------------------------------------

bias_rmse_before_outage = np.sqrt(
    np.mean(
        bias_error_matrix[
            :,
            pre_outage_index
        ]**2
    )
)


bias_rmse_end_outage = np.sqrt(
    np.mean(
        bias_error_matrix[
            :,
            outage_end_index
        ]**2
    )
)


mean_bias_sigma_before_outage = np.mean(
    bias_sigma_matrix[
        :,
        pre_outage_index
    ]
)


mean_true_bias_norm = np.mean(
    np.linalg.norm(
        true_biases,
        axis=1
    )
)


bias_estimation_error_vectors = (
    estimated_biases_before_outage
    - true_biases
)


mean_bias_estimation_error_vector = (
    np.mean(
        bias_estimation_error_vectors,
        axis=0
    )
)


mean_bias_estimation_error_norm = np.linalg.norm(
    mean_bias_estimation_error_vector
)


# ------------------------------------------------------------
# 26. NEES PENDANT OUTAGE
# ------------------------------------------------------------

mean_nees_outage = np.mean(
    nees_matrix[
        :,
        outage_mask
    ]
)


# ------------------------------------------------------------
# 27. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)


print(
    "AURORA — Expérience 008-D"
)


print(
    "Validation Monte Carlo du filtre GNSS + accéléromètre 9D"
)


print(
    "============================================================================================"
)


print(
    f"Nombre de simulations : "
    f"{number_of_runs}"
)


print(
    f"Durée par simulation : "
    f"{simulation_duration_minutes:.1f} min"
)


print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)


print(
    f"Coupure GNSS : "
    f"{gnss_outage_start_minutes:.1f} "
    f"à {gnss_outage_end_minutes:.1f} min"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- NIS -----"
)


print(
    f"Valeur moyenne théorique : "
    f"{measurement_dimension:.3f}"
)


print(
    f"NIS moyen : "
    f"{mean_nis:.3f}"
)


print(
    f"Couverture individuelle 95 % : "
    f"{nis_individual_coverage:.2f} %"
)


print(
    f"Bornes ANIS ensemble 95 % : "
    f"[{anis_lower:.3f}, "
    f"{anis_upper:.3f}]"
)


print(
    f"Epoques ANIS dans les bornes : "
    f"{anis_coverage:.2f} %"
)


print()


print(
    "----- NEES -----"
)


print(
    f"Valeur moyenne théorique : "
    f"{state_dimension:.3f}"
)


print(
    f"NEES moyen : "
    f"{mean_nees:.3f}"
)


print(
    f"NEES moyen pendant outage : "
    f"{mean_nees_outage:.3f}"
)


print(
    f"Couverture individuelle 95 % : "
    f"{nees_individual_coverage:.2f} %"
)


print(
    f"Bornes ANEES ensemble 95 % : "
    f"[{anees_lower:.3f}, "
    f"{anees_upper:.3f}]"
)


print(
    f"Epoques ANEES dans les bornes : "
    f"{anees_coverage:.2f} %"
)


print()


print(
    "----- POSITION -----"
)


print(
    f"RMSE avec GNSS disponible : "
    f"{position_rmse_available:.3f} m"
)


print(
    f"RMSE pendant coupure : "
    f"{position_rmse_outage:.3f} m"
)


print(
    f"RMSE fin de coupure : "
    f"{position_rmse_end_outage:.3f} m"
)


print(
    f"Incertitude 3D moyenne fin coupure : "
    f"{mean_position_sigma_end_outage:.3f} m"
)


print()


print(
    "----- BIAIS ACCELEROMETRE -----"
)


print(
    f"Norme moyenne du biais vrai : "
    f"{mean_true_bias_norm:.6e} m/s^2"
)


print(
    f"RMSE biais avant coupure : "
    f"{bias_rmse_before_outage:.6e} m/s^2"
)


print(
    f"Incertitude biais 3D moyenne "
    f"avant coupure : "
    f"{mean_bias_sigma_before_outage:.6e} m/s^2"
)


print(
    f"RMSE biais fin coupure : "
    f"{bias_rmse_end_outage:.6e} m/s^2"
)


print(
    f"Norme du biais moyen de l'estimation : "
    f"{mean_bias_estimation_error_norm:.6e} m/s^2"
)


print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 28. FIGURE ANEES
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    anees_values,
    label="ANEES"
)


plt.axhline(
    state_dimension,
    linestyle="--",
    label="Valeur théorique = 9"
)


plt.axhline(
    anees_lower,
    linestyle=":"
)


plt.axhline(
    anees_upper,
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
    "AURORA — Cohérence ensemble de l'EKF 9D"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 29. FIGURE ANIS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    anis_values,
    label="ANIS"
)


plt.axhline(
    measurement_dimension,
    linestyle="--",
    label="Valeur théorique = 3"
)


plt.axhline(
    anis_lower,
    linestyle=":"
)


plt.axhline(
    anis_upper,
    linestyle=":"
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
    "ANIS [-]"
)


plt.title(
    "AURORA — Cohérence des innovations GNSS"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 30. FIGURE RMSE POSITION ENSEMBLE
# ------------------------------------------------------------

position_rmse_time = np.sqrt(
    np.mean(
        position_error_matrix**2,
        axis=0
    )
)


mean_position_sigma_time = np.mean(
    position_sigma_matrix,
    axis=0
)


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    position_rmse_time,
    label="RMSE position Monte Carlo"
)


plt.plot(
    time_minutes,
    mean_position_sigma_time,
    linestyle="--",
    label="Incertitude position 3D moyenne"
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
    "Position [m]"
)


plt.title(
    "AURORA — Performance de navigation Monte Carlo"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 31. FIGURE CONVERGENCE DU BIAIS
# ------------------------------------------------------------

bias_rmse_time = np.sqrt(
    np.mean(
        bias_error_matrix**2,
        axis=0
    )
)


mean_bias_sigma_time = np.mean(
    bias_sigma_matrix,
    axis=0
)


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    bias_rmse_time,
    label="RMSE estimation du biais"
)


plt.plot(
    time_minutes,
    mean_bias_sigma_time,
    linestyle="--",
    label="Incertitude biais 3D moyenne"
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
    "AURORA — Convergence Monte Carlo du biais accéléromètre"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()