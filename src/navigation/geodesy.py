import numpy as np


# ============================================================
# AURORA
# Phase 3 - Géodésie
# Conversion ECEF <-> coordonnées géodésiques WGS-84
# ============================================================


# ------------------------------------------------------------
# 1. CONSTANTES WGS-84
# ------------------------------------------------------------

# Demi-grand axe de l'ellipsoïde terrestre [m]
WGS84_A = 6_378_137.0

# Aplatissement
WGS84_F = 1.0 / 298.257223563

# Demi-petit axe [m]
WGS84_B = (
    WGS84_A
    * (1.0 - WGS84_F)
)

# Première excentricité au carré
WGS84_E2 = (
    WGS84_F
    * (2.0 - WGS84_F)
)


# ------------------------------------------------------------
# 2. NORMALISATION DE LA LONGITUDE
# ------------------------------------------------------------

def normalize_longitude(longitude):
    """
    Ramène une longitude dans l'intervalle
    [-pi, pi).

    Parameters
    ----------
    longitude : float
        Longitude [rad].

    Returns
    -------
    float
        Longitude normalisée [rad].
    """

    return (
        longitude + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# ------------------------------------------------------------
# 3. ECEF -> GEODETIQUE
# ------------------------------------------------------------

def ecef_to_geodetic(
    position_ecef,
    tolerance=1e-12,
    max_iterations=20
):
    """
    Convertit une position ECEF en latitude,
    longitude et altitude géodésiques WGS-84.

    Parameters
    ----------
    position_ecef : ndarray
        Position ECEF [X, Y, Z] [m].

    tolerance : float
        Tolérance de convergence [rad].

    max_iterations : int
        Nombre maximal d'itérations.

    Returns
    -------
    latitude : float
        Latitude géodésique [rad].

    longitude : float
        Longitude [rad].

    altitude : float
        Altitude au-dessus de l'ellipsoïde WGS-84 [m].
    """

    x, y, z = position_ecef

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    longitude = np.arctan2(
        y,
        x
    )

    longitude = normalize_longitude(
        longitude
    )

    # --------------------------------------------------------
    # Distance au grand axe de rotation
    # --------------------------------------------------------

    p = np.sqrt(
        x**2 + y**2
    )

    # Cas particulier très proche des pôles
    if p < 1e-10:

        latitude = np.sign(z) * (
            np.pi / 2.0
        )

        altitude = (
            abs(z) - WGS84_B
        )

        return (
            latitude,
            longitude,
            altitude
        )

    # --------------------------------------------------------
    # Première estimation de la latitude
    # --------------------------------------------------------

    latitude = np.arctan2(
        z,
        p * (
            1.0 - WGS84_E2
        )
    )

    # --------------------------------------------------------
    # Raffinement itératif
    # --------------------------------------------------------

    for _ in range(
        max_iterations
    ):

        sin_latitude = np.sin(
            latitude
        )

        prime_vertical_radius = (
            WGS84_A
            /
            np.sqrt(
                1.0
                -
                WGS84_E2
                * sin_latitude**2
            )
        )

        altitude = (
            p / np.cos(latitude)
            -
            prime_vertical_radius
        )

        new_latitude = np.arctan2(
            z,
            p
            * (
                1.0
                -
                WGS84_E2
                * prime_vertical_radius
                / (
                    prime_vertical_radius
                    + altitude
                )
            )
        )

        if (
            abs(
                new_latitude
                - latitude
            )
            < tolerance
        ):

            latitude = new_latitude
            break

        latitude = new_latitude

    # --------------------------------------------------------
    # Altitude finale
    # --------------------------------------------------------

    sin_latitude = np.sin(
        latitude
    )

    prime_vertical_radius = (
        WGS84_A
        /
        np.sqrt(
            1.0
            -
            WGS84_E2
            * sin_latitude**2
        )
    )

    altitude = (
        p / np.cos(latitude)
        -
        prime_vertical_radius
    )

    return (
        latitude,
        longitude,
        altitude
    )


# ------------------------------------------------------------
# 4. GEODETIQUE -> ECEF
# ------------------------------------------------------------

def geodetic_to_ecef(
    latitude,
    longitude,
    altitude
):
    """
    Convertit latitude, longitude et altitude
    en position ECEF selon WGS-84.

    Parameters
    ----------
    latitude : float
        Latitude géodésique [rad].

    longitude : float
        Longitude [rad].

    altitude : float
        Altitude [m].

    Returns
    -------
    ndarray
        Position ECEF [X, Y, Z] [m].
    """

    sin_latitude = np.sin(
        latitude
    )

    cos_latitude = np.cos(
        latitude
    )

    sin_longitude = np.sin(
        longitude
    )

    cos_longitude = np.cos(
        longitude
    )

    prime_vertical_radius = (
        WGS84_A
        /
        np.sqrt(
            1.0
            -
            WGS84_E2
            * sin_latitude**2
        )
    )

    x = (
        prime_vertical_radius
        + altitude
    ) * (
        cos_latitude
        * cos_longitude
    )

    y = (
        prime_vertical_radius
        + altitude
    ) * (
        cos_latitude
        * sin_longitude
    )

    z = (
        prime_vertical_radius
        * (
            1.0 - WGS84_E2
        )
        + altitude
    ) * sin_latitude

    return np.array([
        x,
        y,
        z
    ])