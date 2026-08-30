from pathlib import Path
p=Path('studio/unity/EverestSim/Assets/Scripts/EverestSnowRenderer.cs')
s=p.read_text()
s=s.replace('if (SurfaceKind == rock)', 'if (SurfaceKind == "rock")')
s=s.replace('Shader.SetGlobalFloat(_EverestActiveRadius, 0f);', 'Shader.SetGlobalFloat("_EverestActiveRadius", 0f);')
p.write_text(s)
