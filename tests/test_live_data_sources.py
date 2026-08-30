from __future__ import annotations

import copy
from contextlib import AbstractContextManager
import json
import math
from pathlib import Path
import tempfile
import time

from simulation.data_sources import LiveDataSource
from simulation.live_telemetry import JsonFileTelemetryAdapter, ReplayTelemetryAdapter
from simulation.unity_bridge import ControlRejected, DEFAULT_ENVIRONMENT, UnityRendererBridge


FIXTURE = Path(__file__).parent / "fixtures/live_replay.json"


class raises(AbstractContextManager):
    def __init__(self, error_type, match=None):
        self.error_type = error_type
        self.match = match
        self.value = None

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            raise AssertionError(f"expected {self.error_type.__name__}")
        if not issubclass(error_type, self.error_type):
            return False
        if self.match is not None and self.match not in str(error):
            raise AssertionError(f"{self.match!r} not found in {str(error)!r}")
        self.value = error
        return True


class QueueAdapter:
    kind = "fake"
    name = "fake-sensors"

    def __init__(self, samples):
        self.samples = list(samples)
        self.closed = False

    def poll(self):
        samples, self.samples = self.samples, []
        return samples

    def health(self):
        return {"status": "disconnected" if self.closed else "connected", "last_error": None, "generation": 0}

    def close(self):
        self.closed = True


def robot_sample(sequence=1, sample_time=None, x=1.0):
    return {
        "schema": "everest-live-sample/v1",
        "sample_time": time.time() if sample_time is None else sample_time,
        "robot": {
            "sequence": sequence,
            "body_names": ["pelvis"],
            "body_pos_w": [[x, 2.0, 0.8]],
            "body_quat_w": [[1.0, 0.0, 0.0, 0.0]],
            "joint_names": ["hip"],
            "joint_positions": [0.1],
            "joint_velocities": [0.0],
            "joint_torques": [1.0],
        },
        "provenance": {"robot": "fixture"},
    }


def test_replay_maps_all_live_channels_and_provenance():
    source = LiveDataSource(
        ReplayTelemetryAdapter(FIXTURE),
        default_environment=DEFAULT_ENVIRONMENT,
        stale_after_ms=1000.0,
    )
    frame = source.frame()
    assert frame["sequence"] == 41
    assert frame["body_pos_w"][0] == [1.0, 2.0, 0.82]
    assert frame["body_quat_w"][0] == [1.0, 0.0, 0.0, 0.0]
    assert frame["joint_names"] == ["left_hip_pitch_joint", "right_hip_pitch_joint"]
    environment = source.environment()
    assert math.isclose(environment["wind_speed_m_s"], 20.0)
    assert math.isclose(environment["wind_gust_m_s"], 25.0)
    assert environment["temperature_c"] == -27.0
    assert environment["data_mode"] == "live"
    assert source.terrain()["provenance"] == "robot-lidar-reconstruction"
    assert source.snow()["provenance"] == "foot-probe-reconstruction"
    assert source.sensors()["foot_pressure"]["left_n"] == 280.0
    health = source.health()
    assert health["status"] == "connected"
    assert all(health["channels"][name]["status"] == "connected" for name in ("robot", "weather", "terrain", "snow", "sensors"))
    source.close()


def test_stale_live_frame_is_retained_and_never_replaced():
    sample = robot_sample(x=7.5)
    source = LiveDataSource(
        QueueAdapter([sample]),
        default_environment=DEFAULT_ENVIRONMENT,
        stale_after_ms=5.0,
    )
    assert source.frame()["body_pos_w"][0][0] == 7.5
    time.sleep(0.015)
    assert source.health()["status"] == "stale"
    assert source.frame()["body_pos_w"][0][0] == 7.5


def test_future_and_out_of_order_samples_are_rejected_without_losing_last_good():
    now = time.time()
    adapter = QueueAdapter([robot_sample(2, now, 2.0)])
    source = LiveDataSource(adapter, default_environment=DEFAULT_ENVIRONMENT, stale_after_ms=1000.0)
    assert source.frame()["sequence"] == 2
    adapter.samples.append(robot_sample(3, now + 10.0, 3.0))
    assert source.frame()["sequence"] == 2
    adapter.samples.append(robot_sample(1, now - 1.0, 1.0))
    health = source.health()
    assert source.frame()["sequence"] == 2
    assert health["rejected_samples"] == 2
    assert "non-monotonic" in health["last_error"]


def test_missing_layout_is_reported_in_channel_health():
    source = LiveDataSource(
        QueueAdapter([robot_sample()]),
        default_environment=DEFAULT_ENVIRONMENT,
        stale_after_ms=1000.0,
    )
    source.set_expected_layout(["pelvis", "left_foot"], ["hip", "knee"])
    robot = source.health()["channels"]["robot"]
    assert robot["missing_bodies"] == ["left_foot"]
    assert robot["missing_joints"] == ["knee"]


def test_json_file_adapter_accepts_atomic_snapshot_updates():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "latest.json"
        path.write_text(json.dumps(robot_sample(sequence=1, x=1.0)))
        adapter = JsonFileTelemetryAdapter(path)
        source = LiveDataSource(adapter, default_environment=DEFAULT_ENVIRONMENT, stale_after_ms=1000.0)
        assert source.frame()["sequence"] == 1
        time.sleep(0.02)
        path.write_text(json.dumps(robot_sample(sequence=2, x=2.0)))
        assert source.frame()["sequence"] == 2


class FakeEngine:
    def __init__(self):
        self.controls = []

    def control(self, action, value):
        self.controls.append((action, copy.deepcopy(value)))


def make_control_bridge(live_source):
    bridge = UnityRendererBridge.__new__(UnityRendererBridge)
    bridge.engine = FakeEngine()
    bridge.data_mode = "sim"
    bridge.environment = copy.deepcopy(DEFAULT_ENVIRONMENT)
    bridge.live_source = live_source
    bridge.source_epoch = 1
    bridge._live_generation = 0
    return bridge


def test_live_switch_stops_sim_then_rejects_every_mutation():
    source = LiveDataSource(QueueAdapter([robot_sample()]), default_environment=DEFAULT_ENVIRONMENT)
    bridge = make_control_bridge(source)
    bridge.control("mode", "live")
    assert bridge.data_mode == "live"
    assert bridge.engine.controls[0] == ("command", [0.0, 0.0, 0.0])
    before = list(bridge.engine.controls)
    for action, value in (("command", [1, 0, 0]), ("reset", None), ("pause", True), ("weather", {}), ("surface", "ice")):
        with raises(ControlRejected, match="read-only") as error:
            bridge.control(action, value)
        assert error.value.code == "live_read_only"
    assert bridge.engine.controls == before
    bridge.control("mode", "sim")
    assert bridge.data_mode == "sim"


def test_unconfigured_live_mode_is_rejected_and_sim_remains_active():
    bridge = make_control_bridge(None)
    with raises(ControlRejected) as error:
        bridge.control("mode", "live")
    assert error.value.code == "live_not_configured"
    assert bridge.data_mode == "sim"
    assert bridge.engine.controls == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
