import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# AURORA
# Experiment 001 - Two-Body Orbital Dynamics
# ============================================================

# Earth's gravitational parameter [m^3/s^2]
MU_EARTH = 3.986004418e14

# Mean equatorial radius of Earth [m]
R_EARTH = 6_378_137.0

# Initial spacecraft altitude [m]
ALTITUDE = 550_000.0

# Distance from Earth's center [m]
ORBIT_RADIUS = R_EARTH + ALTITUDE