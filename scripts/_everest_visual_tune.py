from pathlib import Path

for rel in [
    'studio/unity/EverestSim/Assets/Scripts/EverestTerrainRenderer.cs',
    'studio/unity/EverestSim/Assets/Scripts/EverestVisualTerrainRenderer.cs',
]:
    p=Path(rel); s=p.read_text(); s=s.replace('private int _localLodStep = 3;', 'private int _localLodStep = 1;'); p.write_text(s)

p=Path('studio/unity/EverestSim/Assets/Scripts/EverestEditorHud.cs')
s=p.read_text().replace('private int _localLod = 3;', 'private int _localLod = 1;')
p.write_text(s)

p=Path('studio/unity/EverestSim/Assets/Shaders/EverestTerrain.shader')
s=p.read_text()
s=s.replace('_SnowLight ("Snow Light", Color) = (0.95, 0.98, 1.0, 1)', '_SnowLight ("Snow Light", Color) = (0.90, 0.96, 1.0, 1)')
s=s.replace('_SnowShadow ("Snow Shadow", Color) = (0.37, 0.52, 0.68, 1)', '_SnowShadow ("Snow Shadow", Color) = (0.24, 0.42, 0.60, 1)')
s=s.replace('snow *= lerp(0.90.xxx, snowPhoto, 0.46);', 'snow *= lerp(0.86.xxx, snowPhoto, 0.66);\n            snow *= lerp(0.88, 1.08, macro);')
p.write_text(s)

p=Path('studio/unity/EverestSim/Assets/Shaders/EverestSnow.shader')
s=p.read_text()
s=s.replace('_FreshLight ("Fresh Snow Light", Color) = (0.96, 0.99, 1.0, 1)', '_FreshLight ("Fresh Snow Light", Color) = (0.91, 0.97, 1.0, 1)')
s=s.replace('_FreshShadow ("Fresh Snow Shadow", Color) = (0.42, 0.63, 0.82, 1)', '_FreshShadow ("Fresh Snow Shadow", Color) = (0.28, 0.50, 0.70, 1)')
s=s.replace('tex = lerp(1.0.xxx, tex, 0.22);', 'tex = lerp(0.94.xxx, tex, 0.42);')
p.write_text(s)
