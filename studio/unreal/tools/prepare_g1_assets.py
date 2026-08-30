#!/usr/bin/env python3
"""Convert MuJoCo Menagerie G1 STL assets to Unreal-friendly centimeter OBJ meshes."""
from pathlib import Path
import trimesh
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'vendor/mujoco_playground/external_deps/mujoco_menagerie/unitree_g1/assets'
OUT=ROOT/'studio/unreal/EverestSim/SourceData/G1'
OUT.mkdir(parents=True,exist_ok=True)
count=0
for src in sorted(SRC.glob('*.STL')):
    dst=OUT/(src.stem+'.obj')
    mesh=trimesh.load_mesh(src,process=False)
    mesh.apply_scale(100.0)  # Menagerie STL vertices are metres; UE units are centimetres.
    mesh.export(dst)
    count+=1
print(f'wrote {count} G1 OBJ meshes -> {OUT}')
