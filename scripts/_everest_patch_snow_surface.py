from pathlib import Path
p=Path('simulation/snow.py')
s=p.read_text()
old='''    "snow": SnowMaterial("snow", 0.35, 250.0, 2.0e5, 0.30, 2.0e4, 1500.0, (0.82, 0.88, 0.94)),
    "ice": SnowMaterial("ice", 0.08, 917.0, 8.0e6, 0.33, 2.0e5, 0.0, (0.52, 0.72, 0.88)),
    "nominal": SnowMaterial("nominal", 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, (0.10, 0.12, 0.15)),
'''
new='''    "snow": SnowMaterial("snow", 0.35, 250.0, 2.0e5, 0.30, 2.0e4, 1500.0, (0.82, 0.88, 0.94)),
    "ice": SnowMaterial("ice", 0.08, 917.0, 8.0e6, 0.33, 2.0e5, 0.0, (0.52, 0.72, 0.88)),
    "rock": SnowMaterial("rock", 0.82, 2700.0, 5.0e7, 0.25, 2.0e6, 5.0e5, (0.30, 0.29, 0.27)),
    "nominal": SnowMaterial("nominal", 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, (0.10, 0.12, 0.15)),
'''
if old not in s: raise RuntimeError('SURFACES target missing')
s=s.replace(old,new,1)
old='''            "mpm_material": self.surface != "nominal",
            "mpm_ready": self.column is not None,
            "newton_compatible": True,
            "physics_mode": "newton_mpm_pending" if self.column is not None else "mujoco_rigid_friction",
'''
new='''            "mpm_material": self.surface == "snow" and self.column is not None,
            "mpm_ready": self.surface == "snow" and self.column is not None,
            "newton_compatible": True,
            "physics_mode": "newton_mpm_pending" if self.surface == "snow" and self.column is not None else "mujoco_rigid_friction",
'''
if old not in s: raise RuntimeError('manifest target missing')
s=s.replace(old,new,1)
p.write_text(s)
