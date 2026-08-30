# Multilayer Snow Model for Everest Hack

## Goal

Represent Everest snow with the smallest state that is still mechanically useful for Unitree G1 locomotion, incident reconstruction, and Newton Implicit MPM simulation.

Weather provides a prior, robot measurements correct that prior, and incident reconstruction estimates the local material parameters that best reproduce what the robot experienced.

## Architecture

```text
weather + wind + terrain
          |
          v
   predicted snow column
          |
          +---- robot sensors / incident telemetry
          |              |
          v              v
        state estimator / system ID
                  |
                  v
         Newton MPM parameters
                  |
                  v
        MuJoCo G1 <-> Newton snow
```

MuJoCo remains responsible for G1 robot dynamics. Newton Implicit MPM is used for deformable snow and granular contact with two-way coupling.

## Minimum Multilayer State

A snow column contains a small number of mechanically meaningful layers, ordered from the surface downward.

```python
SnowColumn:
    surface_friction: float
    layers: list[SnowLayer]   # top -> bottom

SnowLayer:
    thickness_m: float
    density_kg_m3: float
    stiffness_pa: float
    compressive_strength_pa: float
    shear_strength_pa: float
    compaction_hardening: float
    bond_strength_below_pa: float
```

This is the minimum representation we should expose as estimable state.

### Optional Semantic Type

A semantic type is useful for selecting priors and visualization, but should not replace the numeric mechanical state.

```python
SnowType = POWDER | WIND_PACK | CRUST | DENSE_SNOW | FIRN | ICE
```

## Why These Parameters

### `thickness_m`
Controls the physical extent of the layer and therefore the number and placement of MPM particles.

### `density_kg_m3`
Controls mass, settlement behavior, and part of the expected mechanical regime.

### `stiffness_pa`
Approximate elastic stiffness before significant permanent deformation. Maps primarily to Newton `young_modulus`.

### `compressive_strength_pa`
Controls how much compressive load the layer tolerates before yielding or collapsing. Maps primarily to Newton `yield_pressure`.

### `shear_strength_pa`
Controls resistance to shear deformation inside the layer. Maps primarily to Newton `yield_stress`, together with the chosen internal-friction model.

### `compaction_hardening`
Controls how the material becomes stronger after plastic compaction. Maps to Newton `hardening`.

### `bond_strength_below_pa`
Represents the mechanical strength of the interface between this layer and the layer beneath it. This is critical for weak-layer failure, crust breakthrough, slab-like sliding, and strong-over-weak configurations.

### `surface_friction`
Stored once on `SnowColumn` because it describes G1-foot traction at the exposed surface. It should be treated separately from internal snow friction/rheology.

## Newton Mapping

| Snow state | Newton representation |
|---|---|
| `thickness_m` | MPM particle layer extent |
| `density_kg_m3` | particle density / mass |
| `stiffness_pa` | `mpm.young_modulus` |
| `compressive_strength_pa` | `mpm.yield_pressure` |
| `shear_strength_pa` | `mpm.yield_stress` |
| `compaction_hardening` | `mpm.hardening` |
| internal shear behavior | `mpm.friction` plus strength parameters |
| `surface_friction` | G1/snow collider Coulomb friction |
| `bond_strength_below_pa` | interface-specific coupling/failure approximation |

Newton also exposes Poisson ratio, damping, and tensile-yield ratio. These should remain material-class defaults for the MVP rather than independent estimator variables unless incident data clearly justifies changing them.

## Layer Structure Matters

The model must preserve mechanically distinct layers.

```text
3 cm fresh snow
------------------------
6 cm hard wind crust
========================
25 cm weak loose snow
........................
40 cm dense old snow
########################
glacier ice
========================
```

A single bulk snow material cannot reproduce crust breakthrough, weak-layer collapse, or strong-over-soft support behavior.

## Weather as a Prior

Weather should estimate a plausible initial snow column, not directly predict exact Newton coefficients.

```text
snowfall + temperature
    -> create/add layer
    -> estimate initial density

high wind + terrain exposure
    -> erosion on exposed cells
    -> deposition on lee/concave cells
    -> wind packing
    -> density and stiffness increase

settlement / time
    -> thickness decreases
    -> density increases
    -> stiffness and strength evolve

new snowfall
    -> create a new surface layer
```

The snow state evolves approximately as:

```text
S_t = f(S_(t-1), snowfall, temperature, wind, terrain)
```

## Robot Measurements as Corrections

Weather answers: **what snow do we expect?**

Robot sensing answers: **what snow did we actually experience?**

System identification reconciles the two.

### Foot force and penetration
Force-versus-sinkage provides information about surface-layer stiffness, compressive strength, and compaction hardening. A hard initial response followed by sudden penetration can indicate a crust over a weaker layer.

### Foot slip
Slip onset gives a direct estimate of exposed-surface traction:

```text
surface_friction ~= lateral_force_at_slip / normal_force
```

This should update contact/collider friction, not Newton's internal snow-friction parameter.

### LiDAR / depth sensing
Useful for surface geometry, slope, drift shape, roughness, total visible deformation, and foot penetration before/after contact.

### Radar / GPR, if added later
Can improve total snow depth, layer boundaries, snow-to-ice interfaces, and possible void detection. Radar is not required for the MVP.

## Incident Reconstruction

Snow parameters become part of the latent environment state `theta` already used by Everest Dream incident reconstruction.

```text
find theta
such that
simulated trajectory(theta) ~= recorded incident trajectory
```

Relevant snow components may include surface friction, layer thicknesses, stiffness, compressive strength, shear strength, compaction hardening, and interface/bond strength.

The reconstructed estimate should define a local domain-randomization neighborhood instead of training on one exact fitted value.

## Layer Count

Do not reproduce dozens of microscopic snow layers. For the hackathon, cap the mechanically active representation at approximately **1-6 layers per column**.

Neighboring layers may be merged when density, stiffness, compressive strength, and shear strength are sufficiently similar.

Do **not** merge layers when the interface itself is mechanically important, especially:

- hard crust over weak snow,
- slab over weak layer,
- snow over hard ice,
- strong density/stiffness transitions.

## Spatial Representation

Do not run MPM across the whole mountain. Maintain a lightweight snow-column grid for the route or scene, then instantiate high-resolution Newton MPM only around the robot.

```text
large terrain snow grid
-----------------------------------------

              +-------------+
              | local MPM   |
              | snow patch  |
              |     G1      |
              +-------------+
```

A local patch on the order of 5x5 m to 10x10 m is a reasonable starting point. Rebuild or shift it as the robot moves.

## Ice Handling

Hard glacier ice and rock should generally remain rigid collision geometry with surface friction and roughness rather than MPM material.

Use MPM when the terrain must sink, compact, deform, shear, or collapse. Use rigid geometry when the surface is effectively hard ice or rock.

Accurate brittle glacier-ice fracture or crevasse-bridge failure is outside the MVP unless a dedicated damage/fracture constitutive model is added and calibrated.

## MVP Data Flow

```text
WEATHER API
snow / temperature / wind
          |
          v
Multilayer Snow Estimator
          |
          v
SnowColumn grid
          ^
          |
G1 telemetry / sensing
force / slip / geometry
          |
          v
Incident System Identification
          |
          v
Newton material adapter
          |
          v
MuJoCo G1 <-> Newton MPM snow
```

## Current Dashboard Implementation

The MVP boundary is now implemented in `simulation/snow.py` and
`simulation/newton_snow.py`:

- `POST /api/control` with action `snow_parameters` validates 1-6 layers.
- Layer thickness and density determine particle extent and exact total mass.
- Stiffness, compressive strength, shear strength, hardening, and interface bond
  map to Newton's per-particle constitutive attributes.
- Snowfall is integrated as a mass-conserving surface flux into the fresh top
  layer; wind, temperature, and forecast-driven layer evolution remain priors
  and rendering inputs rather than independent MPM constitutive fields.
- A local tangent-plane MPM patch follows the G1 start area rather than
  simulating the entire Everest tile.
- G1 soles are Newton kinematic colliders. The live MPM height/material grid and
  particles are published at `GET /api/terrain/frame`.
- MuJoCo keeps G1 articulation/contact stability while its local heightfield is
  lowered by Newton's deformed surface. Any resolved Newton sole impulses are
  also forwarded as MuJoCo body wrenches.

The dashboard column is treated as deposited and pre-consolidated. Newton does
not replay gravity-driven deposition from a stress-free lattice on every slider
change; local indentation is initialized from G1 sole pressure and the layer
stress/strain state. This prevents whole-pack free settlement from being
mistaken for robot sinkage.

## Implementation Boundary

Current modules:

```text
simulation/
    snow.py
    newton_snow.py
```

`simulation/snow.py` owns `SnowColumn`, validation, material mapping, and the rigid MuJoCo fallback. Weather-prior updates, sensor corrections, layer creation/merging, and uncertainty metadata remain future work.

`simulation/newton_snow.py` should own `SnowColumn -> Newton` parameter mapping, MPM particle generation by layer, contact/collider friction assignment, and local MPM patch creation/update.

`dashboard/engines/mujoco.py` owns the live dashboard coupling boundary;
`simulation/newton_mujoco.py` remains the standalone Newton-owned rigid-body
bridge and wrench smoke-test boundary.

## MVP Principle

The system does not need a publication-grade forecast of every snow crystal. It needs a compact multilayer mechanical state expressive enough to reproduce the failures that matter to G1 locomotion.

Weather supplies the prior. Robot interaction supplies evidence. System identification finds a plausible local physical state. Newton reproduces that state mechanically for training and incident replay.
