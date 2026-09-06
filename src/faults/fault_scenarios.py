from dataclasses import dataclass, field
from typing import List, Dict
from src.faults.fault_injector import FaultInjector


@dataclass
class ScenarioConfig:
    id: int
    name: str
    description: str
    faults: List[Dict] = field(default_factory=list)


class BenchmarkScenarios:
    """
    Standard benchmark scenarios corresponding to docs/project_definition.md.
    """

    SCENARIOS = {
        1: ScenarioConfig(
            id=1,
            name="Nominal Operation",
            description="All sensors operating cleanly without fault injection.",
            faults=[]
        ),
        2: ScenarioConfig(
            id=2,
            name="GNSS Denial",
            description="Total loss of GNSS signal between t=500s and t=1500s.",
            faults=[
                {"type": "gnss_denial", "t_start": 500.0, "t_end": 1500.0, "params": {}}
            ]
        ),
        3: ScenarioConfig(
            id=3,
            name="IMU Degradation",
            description="IMU noise increases by 20x from t=400s to t=1200s.",
            faults=[
                {"type": "imu_degradation", "t_start": 400.0, "t_end": 1200.0, "params": {"factor": 20.0}}
            ]
        ),
        4: ScenarioConfig(
            id=4,
            name="Star Tracker Failure",
            description="Star tracker blackout from t=600s to t=1400s.",
            faults=[
                {"type": "star_tracker_failure", "t_start": 600.0, "t_end": 1400.0, "params": {}}
            ]
        ),
        5: ScenarioConfig(
            id=5,
            name="GNSS Spoofing",
            description="GNSS position offset of 1000m injected between t=600s and t=1600s.",
            faults=[
                {"type": "gnss_spoofing", "t_start": 600.0, "t_end": 1600.0, "params": {"offset": [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0]}}
            ]
        ),
        6: ScenarioConfig(
            id=6,
            name="Multiple Simultaneous Failures",
            description="GNSS Denial (t=600-1400s) + IMU Degradation (t=800-1200s).",
            faults=[
                {"type": "gnss_denial", "t_start": 600.0, "t_end": 1400.0, "params": {}},
                {"type": "imu_degradation", "t_start": 800.0, "t_end": 1200.0, "params": {"factor": 15.0}}
            ]
        )
    }

    @classmethod
    def get_scenario(cls, scenario_id: int) -> ScenarioConfig:
        if scenario_id not in cls.SCENARIOS:
            raise ValueError(f"Scenario ID {scenario_id} invalid. Choose 1 to 6.")
        return cls.SCENARIOS[scenario_id]

    @classmethod
    def setup_injector(cls, scenario_id: int) -> FaultInjector:
        config = cls.get_scenario(scenario_id)
        injector = FaultInjector()
        for f in config.faults:
            injector.add_fault(
                fault_type=f["type"],
                t_start=f["t_start"],
                t_end=f["t_end"],
                params=f.get("params", {})
            )
        return injector
