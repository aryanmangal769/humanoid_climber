# Everest Studio

The active UI/render path is a native Omniverse Kit / Isaac Sim editor. The old
browser renderer is archived under `archive/web-dashboard-legacy/`.

## What this path owns

- RTX viewport and materials
- USD stage tree, content browser, properties, transform gizmos
- Everest terrain crop and physical XY/Z scaling
- Unitree G1 MJCF -> USD import
- static snowpack parameters stored on `/World/Snow`
- Newton as the active Isaac Sim physics backend

Snow accumulation is intentionally disabled. Snowfall/weather may inform the
initial snapshot, but the simulation starts from a fixed snow column.

## Generate the terrain asset

```bash
.venv-rl/bin/python studio/omniverse/export_terrain.py
```

## Launch

Install or point to Isaac Sim 6.x, then:

```bash
ISAAC_SIM_PATH=/path/to/isaac-sim ./scripts/start-everest-studio.sh
```

The editor uses Kit's native Stage, Property, Content Browser, toolbar and RTX
Viewport instead of recreating them in a web application.

