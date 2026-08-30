from simulation.policy_supervisor import PolicySupervisor, predict_imbalance, route_policy


def test_three_frame_failure_confirmation():
    supervisor = PolicySupervisor()
    risk = predict_imbalance((0.5, 0.0, 0.866), (0.5, 0.0, 0.0), 1)
    assert risk.triggered
    context = {"slope_gradient": 0.1, "friction": 0.2}
    assert not supervisor.observe(risk, context, sim_time=0.02)
    assert not supervisor.observe(risk, context, sim_time=0.04)
    assert supervisor.observe(risk, context, sim_time=0.06)
    assert supervisor.stage == "failure_detected"


def test_nominal_risk_and_deterministic_routes():
    risk = predict_imbalance((0.0, 0.0, 1.0), (0.1, 0.1, 0.0), 2)
    assert not risk.triggered
    assert route_policy({"slope_gradient": 0.01, "friction": 0.6}).requested_key == "flat"
    assert route_policy({"slope_gradient": 0.1, "friction": 0.2}).requested_key == "ice_incline"
    assert route_policy({"slope_gradient": 0.02, "friction": 0.6, "wind_force_n": 10}).requested_key == "wind"


def test_policy_selection_and_execution_are_distinct():
    supervisor = PolicySupervisor()
    supervisor.activate_policy(
        "flat", "Flat-ground walker", "/tmp/policy.onnx", sim_time=0.0
    )
    manifest = supervisor.manifest()
    assert manifest["active_policy_key"] == "flat"
    assert manifest["executed_checkpoint"] == "/tmp/policy.onnx"


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASS {name}")
