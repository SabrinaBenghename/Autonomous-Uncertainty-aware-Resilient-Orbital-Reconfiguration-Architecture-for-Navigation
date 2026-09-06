import numpy as np


class FaultInjector:
    """
    Applies simulated fault modes to sensors during specified time windows.
    Fault Types:
    - 'gnss_denial': Complete GNSS blackout
    - 'gnss_spoofing': Injects incremental/fixed offset into GNSS position
    - 'imu_degradation': Multiplies IMU noise level by degradation factor
    - 'star_tracker_failure': Complete Star Tracker outage
    """

    def __init__(self):
        self.active_faults = []

    def add_fault(
        self,
        fault_type: str,
        t_start: float,
        t_end: float,
        params: dict = None
    ):
        """Register a fault injection event."""
        self.active_faults.append({
            "type": fault_type,
            "t_start": t_start,
            "t_end": t_end,
            "params": params or {}
        })

    def apply_faults(
        self,
        t: float,
        gnss_sensor=None,
        imu_sensor=None,
        star_tracker_sensor=None
    ) -> dict:
        """
        Applies active faults at timestamp t to sensor objects.
        Returns a dict summarizing active fault flags.
        """
        status = {
            "gnss_denied": False,
            "gnss_spoofed": False,
            "imu_degraded": False,
            "star_tracker_failed": False
        }

        # Reset nominal states
        if gnss_sensor is not None:
            gnss_sensor.set_availability(True)
            gnss_sensor.set_spoof_offset(np.zeros(6))

        if imu_sensor is not None:
            imu_sensor.set_degradation_factor(1.0)

        if star_tracker_sensor is not None:
            star_tracker_sensor.set_availability(True)

        # Apply active time window faults
        for fault in self.active_faults:
            if fault["t_start"] <= t <= fault["t_end"]:
                f_type = fault["type"]
                params = fault["params"]

                if f_type == "gnss_denial" and gnss_sensor is not None:
                    gnss_sensor.set_availability(False)
                    status["gnss_denied"] = True

                elif f_type == "gnss_spoofing" and gnss_sensor is not None:
                    offset = params.get("offset", np.array([500.0, 500.0, 500.0, 0.0, 0.0, 0.0]))
                    gnss_sensor.set_spoof_offset(offset)
                    status["gnss_spoofed"] = True

                elif f_type == "imu_degradation" and imu_sensor is not None:
                    factor = params.get("factor", 10.0)
                    imu_sensor.set_degradation_factor(factor)
                    status["imu_degraded"] = True

                elif f_type == "star_tracker_failure" and star_tracker_sensor is not None:
                    star_tracker_sensor.set_availability(False)
                    status["star_tracker_failed"] = True

        return status
