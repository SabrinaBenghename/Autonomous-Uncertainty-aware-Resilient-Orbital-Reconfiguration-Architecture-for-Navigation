import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    propagate_orbit_with_j2_and_drag_like
)

from src.attitude.reference_attitude import (
    build_nadir_pointing_body_to_eci
)

from src.attitude.quaternions import (
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rotation_vector,
    propagate_quaternion_with_body_rate,
    quaternion_attitude_error_angle
)

from src.attitude.attitude_mekf import (
    predict_attitude_mekf,
    update_attitude_mekf_with_star_tracker
)

from src.sensors.gyroscope import (
    simulate_gyroscope_measurement
)

from src.sensors.star_tracker import (
    simulate_star_tracker_measurement
)


# ============================================================
# AURORA
# Experience 011-B
#
# Attitude estimation:
#
#     Gyroscope
#         +
#     Star tracker
#         +
#     gyro bias estimation
#
# via a 6D multiplicative EKF:
#
#     delta_x =
#         [delta_theta,
#          delta_b_g]
#
#
# Comparaison :
#
#     1. gyro-only propagation
#
#     2. MEKF gyro + star tracker
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
# 3. VERITE ORBITALE
# ------------------------------------------------------------

(
    true_initial_position,
    true_initial_velocity
) = keplerian_to_cartesian(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly
)


true_initial_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)


truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            2.0e-5
    )
)


truth_states = (
    truth_solution.y.T
)


maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        -
        time
    )
)


# ------------------------------------------------------------
# 4. ATTITUDE VRAIE
# ------------------------------------------------------------

true_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


true_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


for index in range(
    number_of_epochs
):

    true_rotation_history[
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


    true_quaternion_history[
        index
    ] = (
        rotation_matrix_to_quaternion(
            true_rotation_history[
                index
            ]
        )
    )


# ------------------------------------------------------------
# 5. VITESSE ANGULAIRE VRAIE BODY
# ------------------------------------------------------------

true_body_rate_history = np.zeros(
    (
        number_of_epochs - 1,
        3
    )
)


for index in range(
    number_of_epochs - 1
):

    relative_rotation = (
        true_rotation_history[
            index
        ].T
        @
        true_rotation_history[
            index + 1
        ]
    )


    rotation_vector = (
        rotation_matrix_to_rotation_vector(
            relative_rotation
        )
    )


    true_body_rate_history[
        index
    ] = (
        rotation_vector
        / dt
    )


# ------------------------------------------------------------
# 6. GYROSCOPE
# ------------------------------------------------------------

true_gyro_bias_deg_per_second = np.array([
    0.0010,
    -0.0008,
    0.0006
])


true_gyro_bias_body = np.deg2rad(
    true_gyro_bias_deg_per_second
)


gyro_noise_std_deg_per_second = (
    0.0005
)


gyro_noise_std = np.deg2rad(
    gyro_noise_std_deg_per_second
)


gyro_rng = np.random.default_rng(
    2027
)


gyro_measurements = np.zeros(
    (
        number_of_epochs - 1,
        3
    )
)


for index in range(
    number_of_epochs - 1
):

    result = (
        simulate_gyroscope_measurement(
            true_angular_rate_body=
                true_body_rate_history[
                    index
                ],

            bias_body=
                true_gyro_bias_body,

            noise_std=
                gyro_noise_std,

            rng=
                gyro_rng
        )
    )


    gyro_measurements[
        index
    ] = (
        result[
            "measurement"
        ]
    )


# ------------------------------------------------------------
# 7. STAR TRACKER
# ------------------------------------------------------------

# Synthetic engineering values for this experiment.
#
# These are NOT yet tied to a specific flight sensor.

star_tracker_period_seconds = (
    60.0
)


star_tracker_interval_epochs = int(
    round(
        star_tracker_period_seconds
        / dt
    )
)


star_tracker_noise_std_deg = (
    0.02
)


star_tracker_noise_std_rad = np.deg2rad(
    star_tracker_noise_std_deg
)


star_tracker_rng = np.random.default_rng(
    3030
)


star_tracker_available = np.zeros(
    number_of_epochs,
    dtype=bool
)


star_tracker_measurements = np.full(
    (
        number_of_epochs,
        4
    ),
    np.nan
)


star_tracker_measurement_errors = np.full(
    number_of_epochs,
    np.nan
)


for index in range(
    star_tracker_interval_epochs,
    number_of_epochs,
    star_tracker_interval_epochs
):

    measurement_result = (
        simulate_star_tracker_measurement(
            true_quaternion_body_to_eci=
                true_quaternion_history[
                    index
                ],

            attitude_noise_std_rad=
                star_tracker_noise_std_rad,

            rng=
                star_tracker_rng
        )
    )


    star_tracker_measurements[
        index
    ] = (
        measurement_result[
            "measurement"
        ]
    )


    star_tracker_measurement_errors[
        index
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index
                ],

            estimated_quaternion=
                star_tracker_measurements[
                    index
                ]
        )
    )


    star_tracker_available[
        index
    ] = (
        True
    )


# ------------------------------------------------------------
# 8. GYRO-ONLY INITIALISATION
# ------------------------------------------------------------

gyro_only_quaternion = (
    true_quaternion_history[
        0
    ].copy()
)


gyro_only_attitude_error = np.zeros(
    number_of_epochs
)


# ------------------------------------------------------------
# 9. MEKF INITIALISATION
# ------------------------------------------------------------

mekf_quaternion = (
    true_quaternion_history[
        0
    ].copy()
)


mekf_gyro_bias = np.zeros(
    3
)


initial_attitude_error_sigma_deg = (
    0.10
)


initial_attitude_error_sigma = np.deg2rad(
    initial_attitude_error_sigma_deg
)


initial_gyro_bias_sigma_deg_per_second = (
    0.005
)


initial_gyro_bias_sigma = np.deg2rad(
    initial_gyro_bias_sigma_deg_per_second
)


mekf_covariance = np.diag([
    initial_attitude_error_sigma**2,
    initial_attitude_error_sigma**2,
    initial_attitude_error_sigma**2,

    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2
])


# Synthetic random-walk density.
gyro_bias_random_walk_density_deg = (
    1.0e-7
)


gyro_bias_random_walk_density = np.deg2rad(
    gyro_bias_random_walk_density_deg
)


# ------------------------------------------------------------
# 10. STORAGE
# ------------------------------------------------------------

mekf_attitude_error = np.zeros(
    number_of_epochs
)


mekf_attitude_sigma = np.zeros(
    number_of_epochs
)


mekf_bias_estimate_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


mekf_bias_error_history = np.zeros(
    number_of_epochs
)


mekf_nis = np.full(
    number_of_epochs,
    np.nan
)


# ------------------------------------------------------------
# 11. INITIAL VALUES
# ------------------------------------------------------------

mekf_attitude_sigma[
    0
] = np.sqrt(
    np.trace(
        mekf_covariance[
            0:3,
            0:3
        ]
    )
)


mekf_bias_estimate_history[
    0
] = (
    mekf_gyro_bias
)


mekf_bias_error_history[
    0
] = np.linalg.norm(
    mekf_gyro_bias
    -
    true_gyro_bias_body
)


# ------------------------------------------------------------
# 12. MAIN LOOP
# ------------------------------------------------------------

for index in range(
    number_of_epochs - 1
):

    gyro_measurement = (
        gyro_measurements[
            index
        ]
    )


    # ========================================================
    # A. GYRO-ONLY
    # ========================================================

    gyro_only_quaternion = (
        propagate_quaternion_with_body_rate(
            quaternion_body_to_eci=
                gyro_only_quaternion,

            angular_rate_body=
                gyro_measurement,

            dt=
                dt
        )
    )


    gyro_only_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                gyro_only_quaternion
        )
    )


    # ========================================================
    # B. MEKF PREDICTION
    # ========================================================

    prediction = (
        predict_attitude_mekf(
            quaternion_body_to_eci=
                mekf_quaternion,

            gyro_bias_body=
                mekf_gyro_bias,

            covariance=
                mekf_covariance,

            gyro_measurement_body=
                gyro_measurement,

            dt=
                dt,

            gyro_noise_std=
                gyro_noise_std,

            gyro_bias_random_walk_density=
                gyro_bias_random_walk_density
        )
    )


    predicted_quaternion = (
        prediction[
            "quaternion"
        ]
    )


    predicted_bias = (
        prediction[
            "gyro_bias"
        ]
    )


    predicted_covariance = (
        prediction[
            "covariance"
        ]
    )


    # ========================================================
    # C. STAR TRACKER UPDATE
    # ========================================================

    if star_tracker_available[
        index + 1
    ]:

        update = (
            update_attitude_mekf_with_star_tracker(
                predicted_quaternion=
                    predicted_quaternion,

                predicted_gyro_bias=
                    predicted_bias,

                predicted_covariance=
                    predicted_covariance,

                star_tracker_quaternion=
                    star_tracker_measurements[
                        index + 1
                    ],

                star_tracker_noise_std_rad=
                    star_tracker_noise_std_rad
            )
        )


        mekf_quaternion = (
            update[
                "quaternion"
            ]
        )


        mekf_gyro_bias = (
            update[
                "gyro_bias"
            ]
        )


        mekf_covariance = (
            update[
                "covariance"
            ]
        )


        mekf_nis[
            index + 1
        ] = (
            update[
                "nis"
            ]
        )


    else:

        mekf_quaternion = (
            predicted_quaternion
        )


        mekf_gyro_bias = (
            predicted_bias
        )


        mekf_covariance = (
            predicted_covariance
        )


    # ========================================================
    # D. ERRORS
    # ========================================================

    mekf_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                mekf_quaternion
        )
    )


    mekf_attitude_sigma[
        index + 1
    ] = np.sqrt(
        np.trace(
            mekf_covariance[
                0:3,
                0:3
            ]
        )
    )


    mekf_bias_estimate_history[
        index + 1
    ] = (
        mekf_gyro_bias
    )


    mekf_bias_error_history[
        index + 1
    ] = np.linalg.norm(
        mekf_gyro_bias
        -
        true_gyro_bias_body
    )


# ------------------------------------------------------------
# 13. STATISTICS
# ------------------------------------------------------------

gyro_only_rms_error = np.sqrt(
    np.mean(
        gyro_only_attitude_error**2
    )
)


gyro_only_final_error = (
    gyro_only_attitude_error[
        -1
    ]
)


mekf_rms_error = np.sqrt(
    np.mean(
        mekf_attitude_error**2
    )
)


mekf_final_error = (
    mekf_attitude_error[
        -1
    ]
)


mekf_max_error = np.max(
    mekf_attitude_error
)


valid_star_tracker_errors = np.isfinite(
    star_tracker_measurement_errors
)


star_tracker_rms_measurement_error = np.sqrt(
    np.mean(
        star_tracker_measurement_errors[
            valid_star_tracker_errors
        ]**2
    )
)


valid_nis = np.isfinite(
    mekf_nis
)


mean_nis = np.mean(
    mekf_nis[
        valid_nis
    ]
)


number_of_star_tracker_updates = np.sum(
    star_tracker_available
)


initial_bias_error = (
    mekf_bias_error_history[
        0
    ]
)


final_bias_error = (
    mekf_bias_error_history[
        -1
    ]
)


final_bias_estimate_deg_per_second = np.rad2deg(
    mekf_gyro_bias
)


bias_improvement_factor = (
    initial_bias_error
    /
    final_bias_error
)


# ------------------------------------------------------------
# 14. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================="
)


print(
    "AURORA — Experience 011-B"
)


print(
    "MEKF attitude : gyroscope + star tracker + estimation biais gyro"
)


print(
    "===================================================================================================="
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
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- CAPTEURS -----"
)


print(
    f"Biais gyro vrai BODY : "
    f"{true_gyro_bias_deg_per_second} deg/s"
)


print(
    f"Bruit gyro sigma : "
    f"{gyro_noise_std_deg_per_second:.6f} deg/s"
)


print(
    f"Periode star tracker : "
    f"{star_tracker_period_seconds:.1f} s"
)


print(
    f"Bruit star tracker sigma par axe : "
    f"{star_tracker_noise_std_deg:.4f} deg"
)


print(
    f"Nombre corrections star tracker : "
    f"{number_of_star_tracker_updates}"
)


print()


print(
    "----- GYRO ONLY -----"
)


print(
    f"Erreur attitude RMS : "
    f"{np.rad2deg(gyro_only_rms_error):.3f} deg"
)


print(
    f"Erreur attitude finale : "
    f"{np.rad2deg(gyro_only_final_error):.3f} deg"
)


print()


print(
    "----- STAR TRACKER -----"
)


print(
    f"Erreur mesure attitude RMS : "
    f"{np.rad2deg(star_tracker_rms_measurement_error):.4f} deg"
)


print()


print(
    "----- MEKF -----"
)


print(
    f"Erreur attitude RMS : "
    f"{np.rad2deg(mekf_rms_error):.4f} deg"
)


print(
    f"Erreur attitude finale : "
    f"{np.rad2deg(mekf_final_error):.4f} deg"
)


print(
    f"Erreur attitude maximale : "
    f"{np.rad2deg(mekf_max_error):.4f} deg"
)


print(
    f"Sigma attitude 3D final : "
    f"{np.rad2deg(mekf_attitude_sigma[-1]):.4f} deg"
)


print()


print(
    "----- BIAIS GYRO -----"
)


print(
    f"Erreur biais initiale : "
    f"{np.rad2deg(initial_bias_error):.6f} deg/s"
)


print(
    f"Erreur biais finale : "
    f"{np.rad2deg(final_bias_error):.6f} deg/s"
)


print(
    f"Facteur amelioration biais : "
    f"{bias_improvement_factor:.2f}"
)


print(
    f"Biais estime final : "
    f"{final_bias_estimate_deg_per_second} deg/s"
)


print(
    f"Biais vrai : "
    f"{true_gyro_bias_deg_per_second} deg/s"
)


print()


print(
    "----- COHERENCE -----"
)


print(
    f"NIS moyen star tracker : "
    f"{mean_nis:.3f}"
)


print(
    "Valeur theorique attendue ~3"
)


print(
    "===================================================================================================="
)


# ------------------------------------------------------------
# 15. ATTITUDE ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    np.rad2deg(
        gyro_only_attitude_error
    ),
    label="Gyro only"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_error
    ),
    label="Gyro + star tracker MEKF"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_sigma
    ),
    linestyle="--",
    label="Sigma attitude MEKF 3D"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur attitude [deg]"
)


plt.title(
    "AURORA — Gyro drift vs attitude fusion"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 16. GYRO BIAS ESTIMATION
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_bias_estimate_history[
            :,
            0
        ]
    ),
    label="b_gx estime"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_bias_estimate_history[
            :,
            1
        ]
    ),
    label="b_gy estime"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_bias_estimate_history[
            :,
            2
        ]
    ),
    label="b_gz estime"
)


plt.axhline(
    true_gyro_bias_deg_per_second[
        0
    ],
    linestyle="--"
)


plt.axhline(
    true_gyro_bias_deg_per_second[
        1
    ],
    linestyle="--"
)


plt.axhline(
    true_gyro_bias_deg_per_second[
        2
    ],
    linestyle="--"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Biais gyro [deg/s]"
)


plt.title(
    "AURORA — Estimation du biais gyroscope"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 17. NIS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    mekf_nis,
    marker="o",
    linestyle="-",
    label="NIS star tracker"
)


plt.axhline(
    3.0,
    linestyle="--",
    label="Valeur moyenne theorique = 3"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "NIS [-]"
)


plt.title(
    "AURORA — Coherence statistique du MEKF attitude"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()