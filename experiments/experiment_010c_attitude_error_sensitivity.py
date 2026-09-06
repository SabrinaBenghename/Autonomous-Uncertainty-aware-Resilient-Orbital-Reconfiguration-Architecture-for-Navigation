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

from src.attitude.reference_attitude import (
    build_nadir_pointing_body_to_eci
)

from src.attitude.attitude_errors import (
    apply_body_fixed_attitude_error,
    rotation_error_angle
)

from src.sensors.body_frame_accelerometer import (
    simulate_body_frame_accelerometer_measurement
)

from src.navigation.body_bias_ekf import (
    ekf_predict_with_body_bias_estimation
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_update_position_augmented
)

from src.navigation.gnss_ekf_interface import (
    generate_simplified_gps_positions,
    select_visible_gnss_satellites
)

from src.navigation.gnss_signal_propagation import (
    simulate_pseudoranges_with_transmit_time,
    solve_gnss_least_squares_transmit_time,
    compute_transmit_time_solution_covariance
)


# ============================================================
# AURORA
# Experience 010-C
#
# Sensibilite de la navigation translationnelle
# aux erreurs d'attitude.
#
# Le capteur physique mesure dans le vrai BODY.
#
# Mais le filtre utilise :
#
#     R_BI_est != R_BI_true
#
# pour transformer l'accelerometre vers ECI.
#
#
# Cas testes :
#
#     0.0 deg
#     0.1 deg
#     0.5 deg
#     1.0 deg
#     2.0 deg
#     5.0 deg
#
#
# Erreur appliquee autour de Y_BODY.
#
# Le drag-like force etant principalement
# tangentielle / proche de X_BODY,
# une erreur autour de Y_BODY produit une
# composante parasite hors de cette direction.
# ============================================================


# ------------------------------------------------------------
# 1. CONFIGURATION DES ERREURS D'ATTITUDE
# ------------------------------------------------------------

attitude_error_degrees = np.array([
    0.0,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0
])


number_of_cases = len(
    attitude_error_degrees
)


# Rotation autour de Y_BODY.
attitude_error_axis_body = np.array([
    0.0,
    1.0,
    0.0
])


# ------------------------------------------------------------
# 2. ORBITE
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
# 3. TEMPS
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
    time[
        1
    ]
    - time[
        0
    ]
)


time_minutes = (
    time
    / 60.0
)


# ------------------------------------------------------------
# 4. COUPURE GNSS
# ------------------------------------------------------------

outage_start_minutes = (
    60.0
)

outage_end_minutes = (
    80.0
)


gnss_available = ~(
    (
        time_minutes
        >= outage_start_minutes
    )
    &
    (
        time_minutes
        <= outage_end_minutes
    )
)


outage_mask = (
    ~gnss_available
)


# ------------------------------------------------------------
# 5. VERITE ORBITALE
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


true_non_gravitational_acceleration = (
    2.0e-5
)


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
# 6. ATTITUDE VRAIE
# ------------------------------------------------------------

true_rotation_body_to_eci_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


for index in range(
    number_of_epochs
):

    true_rotation_body_to_eci_history[
        index
    ] = (
        build_nadir_pointing_body_to_eci(
            position_eci=
                truth_states[
                    index,
                    0:3
                ],

            velocity_eci=
                truth_states[
                    index,
                    3:6
                ]
        )
    )


# ------------------------------------------------------------
# 7. ACCELEROMETRE PHYSIQUE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


true_bias_body = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])


accelerometer_rng = np.random.default_rng(
    4242
)


accelerometer_measurements_body = np.zeros(
    (
        number_of_epochs,
        3
    )
)


true_specific_force_body_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


for index in range(
    number_of_epochs
):

    true_specific_force_eci = (
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


    result = (
        simulate_body_frame_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force_eci,

            rotation_body_to_eci=
                true_rotation_body_to_eci_history[
                    index
                ],

            bias_body=
                true_bias_body,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    accelerometer_measurements_body[
        index
    ] = (
        result[
            "measurement_body"
        ]
    )


    true_specific_force_body_history[
        index
    ] = (
        result[
            "true_specific_force_body"
        ]
    )


# ------------------------------------------------------------
# 8. GNSS — PRECALCUL COMMUN
#
# Important :
#
# Tous les cas d'erreur d'attitude recevront
# EXACTEMENT les memes solutions GNSS.
#
# Cela permet d'isoler l'effet de R_BI.
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


gnss_rng = np.random.default_rng(
    2026
)


gnss_positions = np.full(
    (
        number_of_epochs,
        3
    ),
    np.nan
)


gnss_covariances = np.full(
    (
        number_of_epochs,
        3,
        3
    ),
    np.nan
)


gnss_solution_available = np.zeros(
    number_of_epochs,
    dtype=bool
)


gnss_ls_errors = np.full(
    number_of_epochs,
    np.nan
)


pdop_values = np.full(
    number_of_epochs,
    np.nan
)


estimated_clock_bias_meters = (
    0.0
)


for index in range(
    1,
    number_of_epochs
):

    if not gnss_available[
        index
    ]:

        continue


    (
        all_positions,
        all_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[
                index
            ]
    )


    (
        _,
        visible_ids
    ) = select_visible_gnss_satellites(
        receiver_position=
            truth_states[
                index,
                0:3
            ],

        satellite_positions=
            all_positions,

        satellite_ids=
            all_ids
    )


    if len(
        visible_ids
    ) < 4:

        continue


    pseudorange_result = (
        simulate_pseudoranges_with_transmit_time(
            receiver_position=
                truth_states[
                    index,
                    0:3
                ],

            satellite_ids=
                visible_ids,

            reception_time_seconds=
                time[
                    index
                ],

            receiver_clock_bias_seconds=
                receiver_clock_bias_seconds,

            pseudorange_noise_std=
                pseudorange_noise_std,

            rng=
                gnss_rng
        )
    )


    pseudoranges = (
        pseudorange_result[
            "pseudoranges"
        ]
    )


    # --------------------------------------------------------
    # Controlled validation initialisation.
    #
    # Same deterministic offset every epoch.
    # The purpose here is to isolate attitude sensitivity,
    # not GNSS convergence sensitivity.
    # --------------------------------------------------------

    gnss_initial_position = (
        truth_states[
            index,
            0:3
        ]
        +
        np.array([
            1000.0,
            -800.0,
            600.0
        ])
    )


    try:

        solution = (
            solve_gnss_least_squares_transmit_time(
                satellite_ids=
                    visible_ids,

                pseudoranges=
                    pseudoranges,

                reception_time_seconds=
                    time[
                        index
                    ],

                initial_position=
                    gnss_initial_position,

                initial_clock_bias_meters=
                    estimated_clock_bias_meters
            )
        )


        if not solution[
            "converged"
        ]:

            continue


        estimated_clock_bias_meters = (
            solution[
                "clock_bias_meters"
            ]
        )


        gnss_position = (
            solution[
                "position"
            ]
        )


        covariance_result = (
            compute_transmit_time_solution_covariance(
                receiver_position=
                    gnss_position,

                satellite_ids=
                    visible_ids,

                reception_time_seconds=
                    time[
                        index
                    ],

                pseudorange_noise_std=
                    pseudorange_noise_std
            )
        )


        gnss_positions[
            index
        ] = (
            gnss_position
        )


        gnss_covariances[
            index
        ] = (
            covariance_result[
                "position_covariance"
            ]
        )


        gnss_ls_errors[
            index
        ] = np.linalg.norm(
            gnss_position
            - truth_states[
                index,
                0:3
            ]
        )


        pdop_values[
            index
        ] = (
            covariance_result[
                "pdop"
            ]
        )


        gnss_solution_available[
            index
        ] = (
            True
        )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        continue


# ------------------------------------------------------------
# 9. ETAT / COVARIANCE INITIAUX
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


initial_orbital_estimate = (
    true_initial_orbital_state
    +
    np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )
)


initial_state = np.concatenate(
    (
        initial_orbital_estimate,

        np.zeros(
            3
        )
    )
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


initial_covariance = np.diag([
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


bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 10. RESULTATS PAR CAS
# ------------------------------------------------------------

position_error_histories = np.zeros(
    (
        number_of_cases,
        number_of_epochs
    )
)


position_sigma_histories = np.zeros(
    (
        number_of_cases,
        number_of_epochs
    )
)


bias_error_histories = np.zeros(
    (
        number_of_cases,
        number_of_epochs
    )
)


nis_histories = np.full(
    (
        number_of_cases,
        number_of_epochs
    ),
    np.nan
)


measured_attitude_error_deg = np.zeros(
    number_of_cases
)


# ------------------------------------------------------------
# 11. BOUCLE SUR LES ERREURS D'ATTITUDE
# ------------------------------------------------------------

for case_index, error_degrees in enumerate(
    attitude_error_degrees
):

    state = (
        initial_state.copy()
    )


    covariance = (
        initial_covariance.copy()
    )


    # --------------------------------------------------------
    # Etat initial
    # --------------------------------------------------------

    position_error_histories[
        case_index,
        0
    ] = np.linalg.norm(
        state[
            0:3
        ]
        - truth_states[
            0,
            0:3
        ]
    )


    position_sigma_histories[
        case_index,
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    bias_error_histories[
        case_index,
        0
    ] = np.linalg.norm(
        state[
            6:9
        ]
        - true_bias_body
    )


    error_angle_rad = np.deg2rad(
        error_degrees
    )


    # --------------------------------------------------------
    # Verification de l'erreur injectee
    # --------------------------------------------------------

    test_estimated_rotation = (
        apply_body_fixed_attitude_error(
            true_rotation_body_to_eci=
                true_rotation_body_to_eci_history[
                    0
                ],

            error_angle_rad=
                error_angle_rad,

            error_axis_body=
                attitude_error_axis_body
        )
    )


    measured_attitude_error_deg[
        case_index
    ] = np.rad2deg(
        rotation_error_angle(
            true_rotation_body_to_eci=
                true_rotation_body_to_eci_history[
                    0
                ],

            estimated_rotation_body_to_eci=
                test_estimated_rotation
        )
    )


    # --------------------------------------------------------
    # BOUCLE TEMPORELLE
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ====================================================
        # A. ATTITUDE UTILISEE PAR LE FILTRE
        # ====================================================

        estimated_rotation_body_to_eci = (
            apply_body_fixed_attitude_error(
                true_rotation_body_to_eci=
                    true_rotation_body_to_eci_history[
                        index - 1
                    ],

                error_angle_rad=
                    error_angle_rad,

                error_axis_body=
                    attitude_error_axis_body
            )
        )


        # ====================================================
        # B. PREDICTION
        # ====================================================

        (
            predicted_state,
            predicted_covariance
        ) = ekf_predict_with_body_bias_estimation(
            state=
                state,

            covariance=
                covariance,

            dt=
                dt,

            accelerometer_measurement_body=
                accelerometer_measurements_body[
                    index - 1
                ],

            rotation_body_to_eci=
                estimated_rotation_body_to_eci,

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                bias_random_walk_density
        )


        # ====================================================
        # C. GNSS UPDATE
        # ====================================================

        if gnss_solution_available[
            index
        ]:

            update = (
                ekf_update_position_augmented(
                    predicted_state=
                        predicted_state,

                    predicted_covariance=
                        predicted_covariance,

                    position_measurement=
                        gnss_positions[
                            index
                        ],

                    measurement_noise_covariance=
                        gnss_covariances[
                            index
                        ]
                )
            )


            innovation = (
                update[
                    "innovation"
                ]
            )


            innovation_covariance = (
                update[
                    "innovation_covariance"
                ]
            )


            nis_histories[
                case_index,
                index
            ] = (
                innovation.T
                @ np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )


            state = (
                update[
                    "state"
                ]
            )


            covariance = (
                update[
                    "covariance"
                ]
            )


        else:

            state = (
                predicted_state
            )


            covariance = (
                predicted_covariance
            )


        # ====================================================
        # D. ERREURS
        # ====================================================

        position_error_histories[
            case_index,
            index
        ] = np.linalg.norm(
            state[
                0:3
            ]
            - truth_states[
                index,
                0:3
            ]
        )


        position_sigma_histories[
            case_index,
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        bias_error_histories[
            case_index,
            index
        ] = np.linalg.norm(
            state[
                6:9
            ]
            - true_bias_body
        )


# ------------------------------------------------------------
# 12. INDICES IMPORTANT
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]


outage_start_index = (
    outage_indices[
        0
    ]
)


outage_end_index = (
    outage_indices[
        -1
    ]
)


pre_outage_index = (
    outage_start_index
    - 1
)


# ------------------------------------------------------------
# 13. STATISTIQUES PAR CAS
# ------------------------------------------------------------

rmse_gnss_available = np.zeros(
    number_of_cases
)


rmse_outage = np.zeros(
    number_of_cases
)


end_outage_errors = np.zeros(
    number_of_cases
)


end_outage_sigmas = np.zeros(
    number_of_cases
)


pre_outage_bias_errors = np.zeros(
    number_of_cases
)


mean_nis = np.zeros(
    number_of_cases
)


for case_index in range(
    number_of_cases
):

    gnss_mask = (
        gnss_solution_available
    )


    rmse_gnss_available[
        case_index
    ] = np.sqrt(
        np.mean(
            position_error_histories[
                case_index,
                gnss_mask
            ]**2
        )
    )


    rmse_outage[
        case_index
    ] = np.sqrt(
        np.mean(
            position_error_histories[
                case_index,
                outage_mask
            ]**2
        )
    )


    end_outage_errors[
        case_index
    ] = (
        position_error_histories[
            case_index,
            outage_end_index
        ]
    )


    end_outage_sigmas[
        case_index
    ] = (
        position_sigma_histories[
            case_index,
            outage_end_index
        ]
    )


    pre_outage_bias_errors[
        case_index
    ] = (
        bias_error_histories[
            case_index,
            pre_outage_index
        ]
    )


    valid_nis = np.isfinite(
        nis_histories[
            case_index
        ]
    )


    mean_nis[
        case_index
    ] = np.mean(
        nis_histories[
            case_index,
            valid_nis
        ]
    )


# ------------------------------------------------------------
# 14. GNSS STATISTIQUES
# ------------------------------------------------------------

valid_gnss = np.isfinite(
    gnss_ls_errors
)


valid_pdop = np.isfinite(
    pdop_values
)


gnss_ls_rmse = np.sqrt(
    np.mean(
        gnss_ls_errors[
            valid_gnss
        ]**2
    )
)


mean_pdop = np.mean(
    pdop_values[
        valid_pdop
    ]
)


# ------------------------------------------------------------
# 15. FORCE SPECIFIQUE
# ------------------------------------------------------------

mean_specific_force_norm = np.mean(
    np.linalg.norm(
        true_specific_force_body_history,
        axis=1
    )
)


# ------------------------------------------------------------
# 16. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=============================================================================================================="
)


print(
    "AURORA — Experience 010-C"
)


print(
    "Sensibilite de la navigation aux erreurs d'attitude"
)


print(
    "=============================================================================================================="
)


print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
)


print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)


print(
    f"Coupure GNSS : "
    f"{outage_start_minutes:.1f} "
    f"a {outage_end_minutes:.1f} min"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- GNSS COMMUN -----"
)


print(
    f"RMSE GNSS LS : "
    f"{gnss_ls_rmse:.3f} m"
)


print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)


print()


print(
    "----- FORCE SPECIFIQUE -----"
)


print(
    f"Norme moyenne force specifique : "
    f"{mean_specific_force_norm:.6e} m/s^2"
)


print()


print(
    "----- SENSIBILITE ATTITUDE -----"
)


header = (
    f"{'Erreur attitude':>16} | "
    f"{'Biais pre-outage':>17} | "
    f"{'RMSE GNSS':>10} | "
    f"{'RMSE outage':>12} | "
    f"{'Erreur fin':>11} | "
    f"{'Sigma fin':>10} | "
    f"{'NIS':>7}"
)


print(
    header
)


print(
    "-" * len(
        header
    )
)


for case_index in range(
    number_of_cases
):

    print(
        f"{measured_attitude_error_deg[case_index]:15.3f}° | "
        f"{pre_outage_bias_errors[case_index]:17.6e} | "
        f"{rmse_gnss_available[case_index]:10.3f} | "
        f"{rmse_outage[case_index]:12.3f} | "
        f"{end_outage_errors[case_index]:11.3f} | "
        f"{end_outage_sigmas[case_index]:10.3f} | "
        f"{mean_nis[case_index]:7.3f}"
    )


print(
    "=============================================================================================================="
)


# ------------------------------------------------------------
# 17. FIGURE RMSE OUTAGE VS ATTITUDE ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    attitude_error_degrees,
    rmse_outage,
    marker="o"
)


plt.xlabel(
    "Erreur d'attitude [deg]"
)


plt.ylabel(
    "RMSE position pendant coupure [m]"
)


plt.title(
    "AURORA — Sensibilite de la navigation a l'erreur d'attitude"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 18. FIGURE ERREUR FIN OUTAGE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    attitude_error_degrees,
    end_outage_errors,
    marker="o",
    label="Erreur reelle"
)


plt.plot(
    attitude_error_degrees,
    end_outage_sigmas,
    marker="o",
    linestyle="--",
    label="Sigma 3D"
)


plt.xlabel(
    "Erreur d'attitude [deg]"
)


plt.ylabel(
    "Position fin coupure [m]"
)


plt.title(
    "AURORA — Erreur et incertitude en fonction de l'attitude"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 19. FIGURE HISTORIQUES
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)


for case_index, error_degrees in enumerate(
    attitude_error_degrees
):

    plt.plot(
        time_minutes,
        position_error_histories[
            case_index
        ],
        label=f"{error_degrees:.1f} deg"
    )


plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
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
    "AURORA — Propagation des erreurs d'attitude dans la navigation"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()