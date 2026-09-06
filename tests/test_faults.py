import numpy as np
from src.sensors.gnss import GNSSSensor
from src.sensors.imu import IMUSensor
from src.faults.fault_injector import FaultInjector
from src.faults.fault_scenarios import BenchmarkScenarios
from src.faults.gnss_faults import GNSSOutageFault
from src.faults.imu_bias import IMUBiasFault


def test_fault_injector():
    injector = FaultInjector()
    injector.add_fault("gnss_denial", t_start=10.0, t_end=20.0)

    gnss = GNSSSensor()

    # Before fault
    status = injector.apply_faults(5.0, gnss_sensor=gnss)
    assert status["gnss_denied"] is False
    assert gnss.is_available is True

    # During fault
    status = injector.apply_faults(15.0, gnss_sensor=gnss)
    assert status["gnss_denied"] is True
    assert gnss.is_available is False

    # After fault
    status = injector.apply_faults(25.0, gnss_sensor=gnss)
    assert status["gnss_denied"] is False
    assert gnss.is_available is True


def test_gnss_outage_and_imu_bias_faults():
    gnss_fault = GNSSOutageFault()
    gnss_fault.add_outage_window(10.0, 20.0)
    gnss = GNSSSensor()

    res = gnss_fault.evaluate(15.0, gnss)
    assert res["gnss_denied"] is True
    assert gnss.is_available is False

    imu_fault = IMUBiasFault()
    imu_fault.add_degradation_window(10.0, 20.0, factor=5.0)
    imu = IMUSensor()

    res_imu = imu_fault.evaluate(15.0, imu)
    assert res_imu["imu_degraded"] is True
    assert imu.noise_multiplier == 5.0


def test_benchmark_scenarios():
    for scenario_id in range(1, 7):
        scenario = BenchmarkScenarios.get_scenario(scenario_id)
        assert scenario.id == scenario_id
        injector = BenchmarkScenarios.setup_injector(scenario_id)
        assert isinstance(injector, FaultInjector)
