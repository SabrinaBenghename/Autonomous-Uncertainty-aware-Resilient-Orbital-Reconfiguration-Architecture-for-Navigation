import numpy as np
from src.dynamics.orbit_dynamics import OrbitDynamics, R_EARTH, MU_EARTH
from src.dynamics.attitude_dynamics import AttitudeDynamics


def test_circular_velocity():
    alt = 550_000.0
    v_circ = OrbitDynamics.get_circular_velocity(alt)
    expected_v = np.sqrt(MU_EARTH / (R_EARTH + alt))
    np.testing.assert_allclose(v_circ, expected_v, rtol=1e-5)


def test_orbit_propagation_conservation():
    alt = 550_000.0
    r_orbit = R_EARTH + alt
    v_orbit = OrbitDynamics.get_circular_velocity(alt)
    state0 = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

    dynamics = OrbitDynamics(use_j2=False)  # Two-body energy conservation
    t_eval = np.linspace(0, 500, 50)
    states = dynamics.propagate(state0, (0, 500), t_eval)

    e_initial = OrbitDynamics.compute_specific_energy(state0)
    e_final = OrbitDynamics.compute_specific_energy(states[:, -1])

    np.testing.assert_allclose(e_final, e_initial, rtol=1e-4)


def test_attitude_quaternion_normalization():
    att = AttitudeDynamics()
    q_initial = np.array([1.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01])  # q + omega
    t_eval = np.linspace(0, 10, 10)
    states = att.propagate(q_initial, (0, 10), t_eval)

    for i in range(states.shape[1]):
        norm = np.linalg.norm(states[:4, i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)
