import numpy as np


# ============================================================
# AURORA
# Reference spacecraft attitude
#
# Convention:
#
# R_BI maps a vector expressed in BODY coordinates
# into ECI coordinates:
#
#     v_I = R_BI @ v_B
#
# Therefore:
#
#     v_B = R_BI.T @ v_I
#
# The columns of R_BI are the body axes expressed
# in ECI coordinates.
# ============================================================


# ------------------------------------------------------------
# 1. NORMALISATION
# ------------------------------------------------------------

def normalize_vector(
    vector,
    tolerance=1e-15
):
    """
    Retourne un vecteur unitaire.
    """

    vector = np.asarray(
        vector,
        dtype=float
    )


    norm = np.linalg.norm(
        vector
    )


    if norm < tolerance:

        raise ValueError(
            "Impossible de normaliser un vecteur quasi nul."
        )


    return (
        vector
        / norm
    )


# ------------------------------------------------------------
# 2. ATTITUDE NADIR-POINTING
# ------------------------------------------------------------

def build_nadir_pointing_body_to_eci(
    position_eci,
    velocity_eci
):
    """
    Construit une attitude nadir-pointing.

    Convention des axes corps :

        +Z_B : vers le centre de la Terre
               (nadir)

        +X_B : direction tangentielle de l'orbite,
               approximativement dans le sens
               de la vitesse

        +Y_B : complete le repere direct

    La matrice retournee est :

        R_BI

    avec :

        vector_ECI = R_BI @ vector_BODY
    """

    position_eci = np.asarray(
        position_eci,
        dtype=float
    )


    velocity_eci = np.asarray(
        velocity_eci,
        dtype=float
    )


    # --------------------------------------------------------
    # Axe Z corps : nadir
    # --------------------------------------------------------

    radial_unit = normalize_vector(
        position_eci
    )


    body_z_in_eci = (
        -radial_unit
    )


    # --------------------------------------------------------
    # Retirer de la vitesse sa composante selon Z_B
    #
    # On obtient une direction tangentielle locale.
    # --------------------------------------------------------

    tangential_velocity = (
        velocity_eci
        -
        np.dot(
            velocity_eci,
            body_z_in_eci
        )
        * body_z_in_eci
    )


    body_x_in_eci = normalize_vector(
        tangential_velocity
    )


    # --------------------------------------------------------
    # Axe Y pour former un repere direct
    #
    # On veut :
    #
    #     X_B x Y_B = Z_B
    #
    # donc :
    #
    #     Y_B = Z_B x X_B
    # --------------------------------------------------------

    body_y_in_eci = normalize_vector(
        np.cross(
            body_z_in_eci,
            body_x_in_eci
        )
    )


    # --------------------------------------------------------
    # Re-orthogonalisation numerique de X
    # --------------------------------------------------------

    body_x_in_eci = normalize_vector(
        np.cross(
            body_y_in_eci,
            body_z_in_eci
        )
    )


    # --------------------------------------------------------
    # Matrice de rotation BODY -> ECI
    #
    # Chaque colonne est un axe corps exprime
    # dans le repere ECI.
    # --------------------------------------------------------

    rotation_body_to_eci = np.column_stack(
        (
            body_x_in_eci,
            body_y_in_eci,
            body_z_in_eci
        )
    )


    return (
        rotation_body_to_eci
    )


# ------------------------------------------------------------
# 3. BODY -> ECI
# ------------------------------------------------------------

def body_to_eci(
    vector_body,
    rotation_body_to_eci
):
    """
    Transforme un vecteur du repere corps
    vers ECI.
    """

    vector_body = np.asarray(
        vector_body,
        dtype=float
    )


    rotation_body_to_eci = np.asarray(
        rotation_body_to_eci,
        dtype=float
    )


    return (
        rotation_body_to_eci
        @ vector_body
    )


# ------------------------------------------------------------
# 4. ECI -> BODY
# ------------------------------------------------------------

def eci_to_body(
    vector_eci,
    rotation_body_to_eci
):
    """
    Transforme un vecteur ECI
    vers le repere corps.

    Comme une vraie matrice de rotation est
    orthogonale :

        R^-1 = R^T
    """

    vector_eci = np.asarray(
        vector_eci,
        dtype=float
    )


    rotation_body_to_eci = np.asarray(
        rotation_body_to_eci,
        dtype=float
    )


    return (
        rotation_body_to_eci.T
        @ vector_eci
    )


# ------------------------------------------------------------
# 5. VALIDATION DE MATRICE
# ------------------------------------------------------------

def rotation_matrix_diagnostics(
    rotation_matrix
):
    """
    Verifie les proprietes fondamentales
    d'une matrice de rotation.

    Une vraie rotation doit satisfaire :

        R^T R = I

    et :

        det(R) = +1
    """

    rotation_matrix = np.asarray(
        rotation_matrix,
        dtype=float
    )


    identity = np.eye(
        3
    )


    orthogonality_error = np.linalg.norm(
        rotation_matrix.T
        @ rotation_matrix
        - identity,
        ord="fro"
    )


    determinant = np.linalg.det(
        rotation_matrix
    )


    determinant_error = abs(
        determinant
        - 1.0
    )


    return {
        "orthogonality_error":
            orthogonality_error,

        "determinant":
            determinant,

        "determinant_error":
            determinant_error
    }