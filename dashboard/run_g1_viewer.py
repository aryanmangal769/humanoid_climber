"""Open MuJoCo's native viewer on Unitree RL MjLab's canonical G1 scene."""

import mujoco
import mujoco.viewer

from .g1_model import G1_XML


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.793
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    print(f"Launching Unitree MjLab G1: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
