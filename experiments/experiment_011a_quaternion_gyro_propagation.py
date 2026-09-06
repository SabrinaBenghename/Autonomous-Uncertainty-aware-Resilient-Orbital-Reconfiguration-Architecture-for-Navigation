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
    build_nadir_pointing_body_to_eci,
    rotation_matrix_diagnostics
)

from src.attitude.quaternions import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_rotation_vector,
    propagate_quaternion_with_body_rate,
    quaternion_attitude_error_angle
)

from src.sensors.gyroscope import (
    simulate_gyroscope_measurement
)


# ============================================================
# AURORA
# Experience 011-A
#
# Quaternion attitude propagation from gyroscope.
#
# Reference attitude:
#
#     ideal nadir pointing
#
#
# Step 1:
#
#     derive true BODY angular rate from the
#     reference attitude sequence
#
#
# Step 2:
#
#     propagate quaternion with perfect rate
#
#
# Step 3:
#
#     propagate quaternion using a realistic
#     biased/noisy gyroscope
#
#
# Goal:
#
#     validate quaternion convention and show
#     unaided gyro drift.
# ============================================================


# ------------------------------------------------------------
# 1. ORBIT
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
# 2. TIME
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
# 3. ORBITAL TRUTH
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
        - time
    )
)


# ------------------------------------------------------------
# 4. REFERENCE ATTITUDE HISTORY
# ------------------------------------------------------------

reference_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


reference_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


for index in range(
    number_of_epochs
):

    rotation_matrix = (
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


    reference_rotation_history[
        index
    ] = (
        rotation_matrix
    )


    reference_quaternion_history[
        index
    ] = (
        rotation_matrix_to_quaternion(
            rotation_matrix
        )
    )


# ------------------------------------------------------------
# 5. TRUE BODY ANGULAR RATE
#
# Between two epochs:
#
#     R_{k+1} = R_k Delta_R
#
# therefore:
#
#     Delta_R = R_k^T R_{k+1}
#
# Delta_R is the relative rotation expressed
# in the BODY frame at epoch k.
# ------------------------------------------------------------

true_body_rate_history = np.zeros(
    (
        number_of_epochs - 1,
        3
    )
)


relative_rotation_angles = np.zeros(
    number_of_epochs - 1
)


for index in range(
    number_of_epochs - 1
):

    relative_rotation = (
        reference_rotation_history[
            index
        ].T
        @
        reference_rotation_history[
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


    relative_rotation_angles[
        index
    ] = np.linalg.norm(
        rotation_vector
    )


# ------------------------------------------------------------
# 6. GYROSCOPE MODEL
# ------------------------------------------------------------

# Fixed physical gyro bias in BODY.
#
# Values first chosen in deg/s for readability,
# then converted to rad/s.

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


# ------------------------------------------------------------
# 7. PROPAGATED QUATERNIONS
# ------------------------------------------------------------

perfect_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


gyro_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


perfect_quaternion_history[
    0
] = (
    reference_quaternion_history[
        0
    ]
)


gyro_quaternion_history[
    0
] = (
    reference_quaternion_history[
        0
    ]
)


gyro_measurement_history = np.zeros(
    (
        number_of_epochs - 1,
        3
    )
)


perfect_attitude_error = np.zeros(
    number_of_epochs
)


gyro_attitude_error = np.zeros(
    number_of_epochs
)


# ------------------------------------------------------------
# 8. PROPAGATION LOOP
# ------------------------------------------------------------

for index in range(
    number_of_epochs - 1
):

    true_body_rate = (
        true_body_rate_history[
            index
        ]
    )


    # ========================================================
    # A. PERFECT ANGULAR RATE
    # ========================================================

    perfect_quaternion_history[
        index + 1
    ] = (
        propagate_quaternion_with_body_rate(
            quaternion_body_to_eci=
                perfect_quaternion_history[
                    index
                ],

            angular_rate_body=
                true_body_rate,

            dt=
                dt
        )
    )


    # ========================================================
    # B. REALISTIC GYROSCOPE
    # ========================================================

    gyro_result = (
        simulate_gyroscope_measurement(
            true_angular_rate_body=
                true_body_rate,

            bias_body=
                true_gyro_bias_body,

            noise_std=
                gyro_noise_std,

            rng=
                gyro_rng
        )
    )


    gyro_measurement = (
        gyro_result[
            "measurement"
        ]
    )


    gyro_measurement_history[
        index
    ] = (
        gyro_measurement
    )


    gyro_quaternion_history[
        index + 1
    ] = (
        propagate_quaternion_with_body_rate(
            quaternion_body_to_eci=
                gyro_quaternion_history[
                    index
                ],

            angular_rate_body=
                gyro_measurement,

            dt=
                dt
        )
    )


    # ========================================================
    # C. ATTITUDE ERRORS
    # ========================================================

    perfect_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                reference_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                perfect_quaternion_history[
                    index + 1
                ]
        )
    )


    gyro_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                reference_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                gyro_quaternion_history[
                    index + 1
                ]
        )
    )


# ------------------------------------------------------------
# 9. QUATERNION -> ROTATION VALIDATION
# ------------------------------------------------------------

maximum_rotation_reconstruction_error = (
    0.0
)


maximum_rotation_orthogonality_error = (
    0.0
)


maximum_rotation_determinant_error = (
    0.0
)


for index in range(
    number_of_epochs
):

    reconstructed_rotation = (
        quaternion_to_rotation_matrix(
            reference_quaternion_history[
                index
            ]
        )
    )


    reconstruction_error = np.linalg.norm(
        reconstructed_rotation
        -
        reference_rotation_history[
            index
        ],
        ord="fro"
    )


    maximum_rotation_reconstruction_error = max(
        maximum_rotation_reconstruction_error,
        reconstruction_error
    )


    diagnostics = (
        rotation_matrix_diagnostics(
            reconstructed_rotation
        )
    )


    maximum_rotation_orthogonality_error = max(
        maximum_rotation_orthogonality_error,

        diagnostics[
            "orthogonality_error"
        ]
    )


    maximum_rotation_determinant_error = max(
        maximum_rotation_determinant_error,

        diagnostics[
            "determinant_error"
        ]
    )


# ------------------------------------------------------------
# 10. STATISTICS
# ------------------------------------------------------------

true_body_rate_norms = np.linalg.norm(
    true_body_rate_history,
    axis=1
)


mean_body_rate = np.mean(
    true_body_rate_norms
)


minimum_body_rate = np.min(
    true_body_rate_norms
)


maximum_body_rate = np.max(
    true_body_rate_norms
)


maximum_perfect_attitude_error = np.max(
    perfect_attitude_error
)


rms_perfect_attitude_error = np.sqrt(
    np.mean(
        perfect_attitude_error**2
    )
)


final_gyro_attitude_error = (
    gyro_attitude_error[
        -1
    ]
)


maximum_gyro_attitude_error = np.max(
    gyro_attitude_error
)


rms_gyro_attitude_error = np.sqrt(
    np.mean(
        gyro_attitude_error**2
    )
)


gyro_bias_norm = np.linalg.norm(
    true_gyro_bias_body
)


mean_relative_rotation_angle = np.mean(
    relative_rotation_angles
)


# ------------------------------------------------------------
# 11. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================="
)


print(
    "AURORA — Experience 011-A"
)


print(
    "Propagation d'attitude quaternion + gyroscope"
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
    "----- QUATERNION / MATRICE -----"
)


print(
    f"Erreur reconstruction R depuis q max : "
    f"{maximum_rotation_reconstruction_error:.6e}"
)


print(
    f"Erreur orthogonalite max : "
    f"{maximum_rotation_orthogonality_error:.6e}"
)


print(
    f"Erreur determinant max : "
    f"{maximum_rotation_determinant_error:.6e}"
)


print()


print(
    "----- CINEMATIQUE NADIR -----"
)


print(
    f"Vitesse angulaire BODY moyenne : "
    f"{np.rad2deg(mean_body_rate):.6f} deg/s"
)


print(
    f"Vitesse angulaire BODY min : "
    f"{np.rad2deg(minimum_body_rate):.6f} deg/s"
)


print(
    f"Vitesse angulaire BODY max : "
    f"{np.rad2deg(maximum_body_rate):.6f} deg/s"
)


print(
    f"Rotation relative moyenne par pas : "
    f"{np.rad2deg(mean_relative_rotation_angle):.6f} deg"
)


print()


print(
    "----- PROPAGATION PARFAITE -----"
)


print(
    f"Erreur attitude RMS : "
    f"{np.rad2deg(rms_perfect_attitude_error):.9f} deg"
)


print(
    f"Erreur attitude maximale : "
    f"{np.rad2deg(maximum_perfect_attitude_error):.9f} deg"
)


print()


print(
    "----- GYROSCOPE -----"
)


print(
    f"Biais gyro BODY : "
    f"{true_gyro_bias_deg_per_second} deg/s"
)


print(
    f"Norme biais gyro : "
    f"{np.rad2deg(gyro_bias_norm):.6f} deg/s"
)


print(
    f"Bruit gyro sigma par axe : "
    f"{gyro_noise_std_deg_per_second:.6f} deg/s"
)


print(
    f"Erreur attitude RMS avec gyro : "
    f"{np.rad2deg(rms_gyro_attitude_error):.3f} deg"
)


print(
    f"Erreur attitude finale avec gyro : "
    f"{np.rad2deg(final_gyro_attitude_error):.3f} deg"
)


print(
    f"Erreur attitude maximale avec gyro : "
    f"{np.rad2deg(maximum_gyro_attitude_error):.3f} deg"
)


print(
    "===================================================================================================="
)


# ------------------------------------------------------------
# 12. FIGURE BODY RATE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        true_body_rate_history[
            :,
            0
        ]
    ),

    label="omega_x"
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        true_body_rate_history[
            :,
            1
        ]
    ),

    label="omega_y"
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        true_body_rate_history[
            :,
            2
        ]
    ),

    label="omega_z"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Vitesse angulaire BODY [deg/s]"
)


plt.title(
    "AURORA — Vitesse angulaire du repere nadir-pointing"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 13. FIGURE ATTITUDE ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,

    np.rad2deg(
        perfect_attitude_error
    ),

    label="Propagation avec omega parfaite"
)


plt.plot(
    time_minutes,

    np.rad2deg(
        gyro_attitude_error
    ),

    label="Propagation gyro biais + bruit"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur d'attitude [deg]"
)


plt.title(
    "AURORA — Derive d'attitude par integration gyroscopique"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 14. FIGURE GYRO MEASUREMENT
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        gyro_measurement_history[
            :,
            0
        ]
    ),

    label="gyro x"
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        gyro_measurement_history[
            :,
            1
        ]
    ),

    label="gyro y"
)


plt.plot(
    time_minutes[
        :-1
    ],

    np.rad2deg(
        gyro_measurement_history[
            :,
            2
        ]
    ),

    label="gyro z"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Mesure gyro [deg/s]"
)


plt.title(
    "AURORA — Mesures gyroscope simulees"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()