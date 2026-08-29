"""Verify the canonical Unitree RL MjLab G1 scene."""

import json

import mujoco

from .g1_model import G1_XML, validate_g1


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.793
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    result = validate_g1()
    result.update(
        backend="native MuJoCo",
        timestep=float(model.opt.timestep),
        integrator=mujoco.mjtIntegrator(model.opt.integrator).name,
        keyframes=int(model.nkey),
        note="Unitree RL MjLab scene loaded and reset to a standing viewer pose.",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
