import numpy as np


# ============================================================
# AURORA
# Quaternion utilities
#
# Convention:
#
#     q_BI = [w, x, y, z]
#
# represents the rotation:
#
#     BODY -> ECI
#
# so:
#
#     v_I = R_BI @ v_B
#
#
# Quaternion propagation:
#
#     q_{k+1} = q_k ⊗ delta_q
#
# where delta_q represents the body rotation during dt
# expressed relative to the BODY frame.
# ============================================================


# ------------------------------------------------------------
# 1. NORMALISATION
# ------------------------------------------------------------

def normalize_quaternion(
    quaternion,
    tolerance=1e-15
):
    """
    Normalise un quaternion [w, x, y, z].
    """

    quaternion = np.asarray(
        quaternion,
        dtype=float
    )


    norm = np.linalg.norm(
        quaternion
    )


    if norm < tolerance:

        raise ValueError(
            "Impossible de normaliser un quaternion quasi nul."
        )


    return (
        quaternion
        / norm
    )


# ------------------------------------------------------------
# 2. PRODUIT DE HAMILTON
# ------------------------------------------------------------

def quaternion_multiply(
    quaternion_1,
    quaternion_2
):
    """
    Produit de Hamilton :

        q = q1 ⊗ q2
    """

    q1 = normalize_quaternion(
        quaternion_1
    )


    q2 = normalize_quaternion(
        quaternion_2
    )


    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2


    return np.array([
        (
            w1 * w2
            - x1 * x2
            - y1 * y2
            - z1 * z2
        ),

        (
            w1 * x2
            + x1 * w2
            + y1 * z2
            - z1 * y2
        ),

        (
            w1 * y2
            - x1 * z2
            + y1 * w2
            + z1 * x2
        ),

        (
            w1 * z2
            + x1 * y2
            - y1 * x2
            + z1 * w2
        )
    ])


# ------------------------------------------------------------
# 3. QUATERNION -> MATRICE DE ROTATION
# ------------------------------------------------------------

def quaternion_to_rotation_matrix(
    quaternion
):
    """
    Convertit q_BI en R_BI.

    Convention :

        v_I = R_BI @ v_B
    """

    q = normalize_quaternion(
        quaternion
    )


    w, x, y, z = q


    return np.array([
        [
            1.0 - 2.0 * (y**2 + z**2),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y)
        ],

        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x**2 + z**2),
            2.0 * (y * z - w * x)
        ],

        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x**2 + y**2)
        ]
    ])


# ------------------------------------------------------------
# 4. MATRICE -> QUATERNION
# ------------------------------------------------------------

def rotation_matrix_to_quaternion(
    rotation_matrix
):
    """
    Convertit une matrice R_BI en quaternion
    [w, x, y, z].
    """

    R = np.asarray(
        rotation_matrix,
        dtype=float
    )


    if R.shape != (3, 3):

        raise ValueError(
            "La matrice de rotation doit etre 3x3."
        )


    trace = np.trace(
        R
    )


    if trace > 0.0:

        s = np.sqrt(
            trace + 1.0
        ) * 2.0


        w = (
            0.25
            * s
        )


        x = (
            R[2, 1]
            - R[1, 2]
        ) / s


        y = (
            R[0, 2]
            - R[2, 0]
        ) / s


        z = (
            R[1, 0]
            - R[0, 1]
        ) / s


    elif (
        R[0, 0] > R[1, 1]
        and
        R[0, 0] > R[2, 2]
    ):

        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0


        w = (
            R[2, 1]
            - R[1, 2]
        ) / s


        x = (
            0.25
            * s
        )


        y = (
            R[0, 1]
            + R[1, 0]
        ) / s


        z = (
            R[0, 2]
            + R[2, 0]
        ) / s


    elif R[1, 1] > R[2, 2]:

        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0


        w = (
            R[0, 2]
            - R[2, 0]
        ) / s


        x = (
            R[0, 1]
            + R[1, 0]
        ) / s


        y = (
            0.25
            * s
        )


        z = (
            R[1, 2]
            + R[2, 1]
        ) / s


    else:

        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0


        w = (
            R[1, 0]
            - R[0, 1]
        ) / s


        x = (
            R[0, 2]
            + R[2, 0]
        ) / s


        y = (
            R[1, 2]
            + R[2, 1]
        ) / s


        z = (
            0.25
            * s
        )


    return normalize_quaternion(
        np.array([
            w,
            x,
            y,
            z
        ])
    )


# ------------------------------------------------------------
# 5. ROTATION VECTOR -> QUATERNION
# ------------------------------------------------------------

def rotation_vector_to_quaternion(
    rotation_vector
):
    """
    Convertit un rotation vector :

        delta_theta = axis * angle

    en quaternion.
    """

    rotation_vector = np.asarray(
        rotation_vector,
        dtype=float
    )


    angle = np.linalg.norm(
        rotation_vector
    )


    if angle < 1e-14:

        return normalize_quaternion(
            np.array([
                1.0,
                0.5 * rotation_vector[0],
                0.5 * rotation_vector[1],
                0.5 * rotation_vector[2]
            ])
        )


    axis = (
        rotation_vector
        / angle
    )


    half_angle = (
        0.5
        * angle
    )


    return np.concatenate(
        (
            np.array([
                np.cos(
                    half_angle
                )
            ]),

            axis
            * np.sin(
                half_angle
            )
        )
    )


# ------------------------------------------------------------
# 6. MATRICE -> ROTATION VECTOR
# ------------------------------------------------------------

def rotation_matrix_to_rotation_vector(
    rotation_matrix
):
    """
    Calcule le rotation vector associe a une
    matrice de rotation.

    Pour notre cas, les rotations entre deux
    epoques de 10 s sont petites.
    """

    R = np.asarray(
        rotation_matrix,
        dtype=float
    )


    cosine_angle = (
        (
            np.trace(
                R
            )
            - 1.0
        )
        / 2.0
    )


    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )


    angle = np.arccos(
        cosine_angle
    )


    if angle < 1e-12:

        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])


    sine_angle = np.sin(
        angle
    )


    axis = (
        1.0
        /
        (
            2.0
            * sine_angle
        )
        *
        np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
    )


    return (
        angle
        * axis
    )


# ------------------------------------------------------------
# 7. PROPAGATION QUATERNION PAR GYRO
# ------------------------------------------------------------

def propagate_quaternion_with_body_rate(
    quaternion_body_to_eci,
    angular_rate_body,
    dt
):
    """
    Propagation discrete :

        delta_theta = omega_B * dt

        q_{k+1}
            =
        q_k ⊗ delta_q

    omega_B :
        vitesse angulaire du BODY par rapport
        a l'inertiel, exprimee dans BODY.
    """

    q = normalize_quaternion(
        quaternion_body_to_eci
    )


    angular_rate_body = np.asarray(
        angular_rate_body,
        dtype=float
    )


    rotation_vector = (
        angular_rate_body
        * dt
    )


    delta_quaternion = (
        rotation_vector_to_quaternion(
            rotation_vector
        )
    )


    propagated_quaternion = (
        quaternion_multiply(
            q,
            delta_quaternion
        )
    )


    return normalize_quaternion(
        propagated_quaternion
    )


# ------------------------------------------------------------
# 8. ERREUR ANGULAIRE ENTRE DEUX QUATERNIONS
# ------------------------------------------------------------

def quaternion_attitude_error_angle(
    reference_quaternion,
    estimated_quaternion
):
    """
    Retourne l'angle principal d'erreur
    d'attitude [rad].

    q et -q representent la meme attitude,
    donc on utilise la valeur absolue du
    produit scalaire.
    """

    q_reference = normalize_quaternion(
        reference_quaternion
    )


    q_estimated = normalize_quaternion(
        estimated_quaternion
    )


    scalar_product = abs(
        np.dot(
            q_reference,
            q_estimated
        )
    )


    scalar_product = np.clip(
        scalar_product,
        -1.0,
        1.0
    )


    return (
        2.0
        * np.arccos(
            scalar_product
        )
    )