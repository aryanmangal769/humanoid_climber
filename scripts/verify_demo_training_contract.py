#!/usr/bin/env python3
"""Fail builds that regress the autonomous demo training viewport contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUD = ROOT / "studio/unity/EverestSim/Assets/Scripts/EverestEditorHud.cs"
CLIENT = ROOT / "studio/unity/EverestSim/Assets/Scripts/EverestBackendClient.cs"


def require(source: str, marker: str, path: Path) -> None:
    if marker not in source:
        raise SystemExit(f"demo training contract missing {marker!r} in {path}")


def main() -> None:
    hud = HUD.read_text()
    client = CLIENT.read_text()
    for marker in (
        'Value<bool?>("training_view_active") == true',
        "DrawDemoTrainingFullscreen",
        '"SKIP RL PHASE"',
        '"STOP JOURNEY"',
        "_backend.SendDemoSkipPhase()",
        "_backend.SendDemoStop(!stopped)",
        "_backend.SendSubsetPreview(true)",
    ):
        require(hud, marker, HUD)
    if 'Value<bool?>("training_view_active") == true &&' in hud:
        raise SystemExit(
            "demo training contract regressed: fullscreen RL is gated by stale local UI state"
        )
    require(client, 'SendControl("demo_skip_phase"', CLIENT)
    require(client, 'SendControl("demo_stop"', CLIENT)
    for forbidden in (
        "Autonomous showcase is presentation-only",
        "The autonomous showcase has no RL operator surface",
    ):
        if forbidden in hud:
            raise SystemExit(
                "demo training contract regressed: force-close showcase lifecycle is present"
            )
    print("demo-training-contract: PASS")


if __name__ == "__main__":
    main()
