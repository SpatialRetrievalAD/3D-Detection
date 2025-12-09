import os, json, pickle, sys
from collections import defaultdict

# ====== hardcode ======
NUSC_DATAROOT = "xxx/dataset/nuscenes"
GEOEXT_ROOT   = "xxx/dataset/streetdata"
NUSC_VERSION  = "v1.0-trainval"           
OUT_PKL       = f"xxx/dataset/nuscenes/nus_geoext.pkl"

# same as cfg
CAM_CHANNELS = [
    "CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT",
]

from nuscenes.nuscenes import NuScenes
from nuscenes_geoext.nugeo import NuScenesGeoExt

def _extract_paths_from_geoext_record(rec: dict):
    topdown_path = None
    for k in ["sate_path", "satellite_path", "topdown_path"]:
        v = rec.get(k)
        if isinstance(v, str):
            topdown_path = v
            break

    slice6_paths = None
    for k in ["sate_slice_paths", "sat_slice_paths", "slice_paths"]:
        v = rec.get(k)
        if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            slice6_paths = list(v)[:6]
            break

    return topdown_path, slice6_paths

def main():

    print("== init NuScenes / GeoExt ==")
    nusc  = NuScenes(dataroot=NUSC_DATAROOT, version=NUSC_VERSION, verbose=True)
    nugeo = NuScenesGeoExt(
        dataroot=NUSC_DATAROOT,
        version=NUSC_VERSION,
        geoext_dataroot=GEOEXT_ROOT,
        streetview_fov=60,
        streetview_size=(448, 256),
        sate_size=(400, 400),
    )
    
if __name__ == "__main__":
    main()
