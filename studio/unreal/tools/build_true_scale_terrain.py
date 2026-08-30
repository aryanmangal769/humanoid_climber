#!/usr/bin/env python3
"""Build true-scale macro and local Everest DEM manifests around the South Col route."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
DEM=ROOT/'maps/output_hh.tif'; ROUTE=ROOT/'maps/export.geojson'; OUT=ROOT/'studio/unreal/config'
WEST,EAST=86.65,86.95; SOUTH,NORTH=27.63,28.05
LOCAL_SIZE_M=1200.0; LOCAL_RES=257; MACRO_W=241

def bilinear(grid, lon, lat):
    rows,cols=grid.shape
    x=np.clip((lon-WEST)/(EAST-WEST)*(cols-1),0,cols-1)
    y=np.clip((NORTH-lat)/(NORTH-SOUTH)*(rows-1),0,rows-1)
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int); x1=np.minimum(x0+1,cols-1); y1=np.minimum(y0+1,rows-1)
    fx=x-x0; fy=y-y0
    return grid[y0,x0]*(1-fx)*(1-fy)+grid[y0,x1]*fx*(1-fy)+grid[y1,x0]*(1-fx)*fy+grid[y1,x1]*fx*fy

def manifest(name, heights, width_m, depth_m, center, anchor_elev, route=None, source=''):
    return {'schema':'everest-terrain/v1','name':name,'grid_width':int(heights.shape[1]),'grid_height':int(heights.shape[0]),
            'world_width_m':float(width_m),'world_depth_m':float(depth_m),'terrain_center':[float(center[0]),float(center[1]),0.0],
            'vertical_relief_m':float(heights.max()-heights.min()),'route_start_elevation_m':float(anchor_elev),
            'min_elevation_m':float(heights.min()+anchor_elev),'max_elevation_m':float(heights.max()+anchor_elev),
            'heights':[round(float(v),5) for v in heights.ravel()], 'route':route or [], 'source':source, 'true_scale':True}

grid=np.asarray(Image.open(DEM),dtype=np.float64)
feature=json.loads(ROUTE.read_text())['features'][0]; coords=feature['geometry']['coordinates']; anchor_lon,anchor_lat=coords[0]
center_lat=(NORTH+SOUTH)/2; mx=111_320*math.cos(math.radians(center_lat)); my=111_320
anchor_elev=float(bilinear(grid,np.array(anchor_lon),np.array(anchor_lat)))
route_xyz=[[(lon-anchor_lon)*mx,(lat-anchor_lat)*my,float(bilinear(grid,np.array(lon),np.array(lat))-anchor_elev)+0.15] for lon,lat in coords]
# Macro: measured full DEM, true metres, downsampled only for rendering cost.
macro_h=round(MACRO_W*grid.shape[0]/grid.shape[1]); macro=np.asarray(Image.fromarray(grid.astype(np.float32),mode='F').resize((MACRO_W,macro_h),Image.Resampling.BILINEAR),dtype=np.float64)-anchor_elev
macro_width=(EAST-WEST)*mx; macro_depth=(NORTH-SOUTH)*my
macro_center=(((WEST+EAST)/2-anchor_lon)*mx,((SOUTH+NORTH)/2-anchor_lat)*my)
(OUT/'everest_macro_terrain.json').write_text(json.dumps(manifest('Everest macro DEM',macro,macro_width,macro_depth,macro_center,anchor_elev,route_xyz,'maps/output_hh.tif'),separators=(',',':')))
# Local: exact true-scale crop around G1, resampled from the same measured DEM.
half=LOCAL_SIZE_M/2; xs=np.linspace(-half,half,LOCAL_RES); ys=np.linspace(half,-half,LOCAL_RES) # north -> south
lon=anchor_lon+xs[None,:]/mx; lat=anchor_lat+ys[:,None]/my
local=bilinear(grid,lon,lat)-anchor_elev
(OUT/'everest_robot_terrain.json').write_text(json.dumps(manifest('Everest G1 local DEM',local,LOCAL_SIZE_M,LOCAL_SIZE_M,(0,0),anchor_elev,[[x,y,z] for x,y,z in route_xyz if abs(x)<=half and abs(y)<=half],'maps/output_hh.tif'),separators=(',',':')))
print(f'anchor {anchor_lon:.7f},{anchor_lat:.7f} elev={anchor_elev:.1f}m')
print(f'local {LOCAL_RES}x{LOCAL_RES} {LOCAL_SIZE_M:.0f}m relief={local.max()-local.min():.1f}m')
print(f'macro {MACRO_W}x{macro_h} {macro_width/1000:.1f}x{macro_depth/1000:.1f}km')
