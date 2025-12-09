import os
import pickle
from io import BytesIO

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from PIL.Image import Image as PILImage

# ======== hardcode ========
NUSC_VERSION   = 'v1.0-trainval'
NUSC_DATAROOT  = 'data/nuscenes'
EXTRA_TAG      = 'bevdetv3-nuscenes'

# bevdet pkl
BEVDET_TRAIN_PKL = os.path.join(NUSC_DATAROOT, f'{EXTRA_TAG}_infos_train.pkl')
BEVDET_VAL_PKL   = os.path.join(NUSC_DATAROOT, f'{EXTRA_TAG}_infos_val.pkl')

# geoext pkl
STREETVIEW_PKL   = 'data/streetview_bevdet_geo.pkl'
SATELLITE_PKL    = 'data/satellite_bevdet_geo.pkl'

OUT_SUFFIX = '_geoext'

EXTRINSIC_MODE = 'cam2ego'  # (pano/cam -> ego)

SAT_SCALE_M_PER_PX = 0.15

CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]

VIRTUAL_K = np.array([
    [553.4595,   0.0000, 352.0000],
    [  0.0000, 553.4595, 128.0000],
    [  0.0000,   0.0000,   1.0000]
], dtype=np.float32)

def _ego2lidar_from_info(info):
    R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    t_l2e = np.array(info['lidar2ego_translation'], dtype=np.float32)
    R_e2l = R_l2e.T
    t_e2l = - R_e2l @ t_l2e
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_e2l
    T[:3, 3]  = t_e2l
    return T

def _make_img2lidar(E_c_in, mode, info):
    T_LE = _ego2lidar_from_info(info)  # ^L T_E
    if mode != 'cam2ego':
        raise ValueError('EXTRINSIC_MODE invalid')
    T_EC = np.asarray(E_c_in, dtype=np.float32)
    return (T_LE @ T_EC).astype(np.float32)

def _pix2lidar_from_size(size_wh, scale=SAT_SCALE_M_PER_PX):
    W, H = size_wh
    s = float(scale)
    cx, cy = W / 2.0, H / 2.0
    M = np.array([
        [0.0, -s,  s * cy],
        [-s,  0.0, s * cx],
        [0.0,  0.0, 1.0]
    ], dtype=np.float32)
    return M

def _size_from_bytes(img_bytes):
    w, h = Image.open(BytesIO(img_bytes)).size
    return (int(w), int(h))

def _read_bytes_maybe(path_or_bytes, geoext_root=None):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return bytes(path_or_bytes)
    if isinstance(path_or_bytes, str):
        p = path_or_bytes if (geoext_root is None or os.path.isabs(path_or_bytes)) \
            else os.path.join(geoext_root, path_or_bytes)
        with open(p, 'rb') as f:
            return f.read()
    if isinstance(path_or_bytes, PILImage):
        buf = BytesIO()
        path_or_bytes.save(buf, format='PNG')
        return buf.getvalue()
    return None

def _get_cam_sd_token_from_info(nusc, info, cam_name):
    cam = info['cams'][cam_name]
    if 'sample_data_token' in cam:
        return cam['sample_data_token']
    try:
        sample = nusc.get('sample', info['token'])
        return sample['data'][cam_name]
    except Exception:
        return None

def _get_lidar_sd_token_from_info(nusc, info):
    for k in ['lidar_token', 'lidar_sample_data_token', 'lidar_sd_token']:
        if k in info:
            return info[k]
    try:
        sample = nusc.get('sample', info['token'])
        return sample['data']['LIDAR_TOP']
    except Exception:
        return None

def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _as_4x4(E):
    E = np.asarray(E, dtype=np.float32)
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = E
        return T
    if E.shape == (3, 3):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = E
        return T
    return None

def _finite_or_identity(E):
    if E is None:
        return np.eye(4, dtype=np.float32)
    E4 = _as_4x4(E)
    if E4 is None:
        return np.eye(4, dtype=np.float32)
    if not np.all(np.isfinite(E4)):
        return np.eye(4, dtype=np.float32)
    return E4

def _K3x3_from_any(K):
    if K is None:
        return VIRTUAL_K.copy()
    K = np.asarray(K, dtype=np.float32)
    if K.shape == (3, 3):
        return K
    if K.shape == (4, 4):
        return K[:3, :3]
    # 也支持 [fx, fy, cx, cy]
    if K.size == 4:
        fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
        fx = fx if np.isfinite(fx) and fx > 1e-6 else VIRTUAL_K[0, 0]
        fy = fy if np.isfinite(fy) and fy > 1e-6 else VIRTUAL_K[1, 1]
        cx = cx if np.isfinite(cx) else VIRTUAL_K[0, 2]
        cy = cy if np.isfinite(cy) else VIRTUAL_K[1, 2]
        K3 = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        return K3
    return VIRTUAL_K.copy()

def _fxfycxcy_from_K(K3):
    fx = float(K3[0, 0]); fy = float(K3[1, 1]); cx = float(K3[0, 2]); cy = float(K3[1, 2])
    fx = fx if np.isfinite(fx) and fx > 1e-6 else float(VIRTUAL_K[0, 0])
    fy = fy if np.isfinite(fy) and fy > 1e-6 else float(VIRTUAL_K[1, 1])
    cx = cx if np.isfinite(cx) else float(VIRTUAL_K[0, 2])
    cy = cy if np.isfinite(cy) else float(VIRTUAL_K[1, 2])
    return [fx, fy, cx, cy]

def _merge_one(bevdet_pkl_path, out_suffix, nusc, stv_db, sat_db, geoext_root_hint=None):
    if not os.path.exists(bevdet_pkl_path):
        print(f'[skip] not found: {bevdet_pkl_path}')
        return

    data = _load_pkl(bevdet_pkl_path)
    infos = data['infos']
    print(f'[load] {bevdet_pkl_path}, infos={len(infos)}')

    geoext_root = geoext_root_hint
    if geoext_root is None:
        for v in stv_db.values():
            p = v.get('img_path') or v.get('streetview_img') or v.get('streetview_path')
            if isinstance(p, str):
                geoext_root = os.path.abspath(os.path.join(p, '..', '..')) if not os.path.isabs(p) else '/'
                break

    miss_stv, miss_sat = 0, 0
    for idx, info in enumerate(infos):
        stv_img2lidar = {}
        stv_img_bytes = {}
        stv_K = {}
        stv_E_cam2lidar = {}
        stv_fxfycxcy = {}

        for cam_name in CAMERAS:
            if cam_name not in info.get('cams', {}):
                continue
            cam_sd_token = _get_cam_sd_token_from_info(nusc, info, cam_name)
            if cam_sd_token is None or cam_sd_token not in stv_db:
                miss_stv += 1
                continue
            rec = stv_db[cam_sd_token]

            E_raw = rec.get('extrinsic') or rec.get('streetview_extrinsic') or rec.get('E')
            K_raw = (rec.get('intrinsic') or rec.get('cam_intrinsic')
                     or rec.get('K') or rec.get('streetview_intrinsic'))

            E = _finite_or_identity(E_raw)      # ^E T_C (cam2ego) 4x4
            K3 = _K3x3_from_any(K_raw)          # 3x3
            fxfycxcy = _fxfycxcy_from_K(K3)

            stv_K[cam_name] = K3.astype(np.float32)
            stv_fxfycxcy[cam_name] = np.asarray(fxfycxcy, dtype=np.float32)

            try:
                T_LC = _make_img2lidar(E, EXTRINSIC_MODE, info).astype(np.float32)  # 4x4
                stv_E_cam2lidar[cam_name] = T_LC.astype(np.float32)
                T_CL = np.linalg.inv(T_LC)  # 4x4
                R_CL = T_CL[:3, :3]
                t_CL = T_CL[:3, 3:4]        # (3,1)

                P = K3 @ np.concatenate([R_CL, t_CL], axis=1)  # (3,4)

                lidar2img = np.eye(4, dtype=np.float32)
                lidar2img[:3, :4] = P.astype(np.float32)

                img2lidar = np.linalg.inv(lidar2img).astype(np.float32)

                if np.all(np.isfinite(img2lidar)):
                    stv_img2lidar[cam_name] = img2lidar
            except Exception:
                miss_stv += 1

            img_obj = rec.get('img_bytes') or rec.get('streetview_img') or rec.get('img') or rec.get('img_path')
            b = _read_bytes_maybe(img_obj, geoext_root)
            if b is not None:
                stv_img_bytes[cam_name] = b

        if stv_img2lidar:
            info['streetview_img2lidar'] = stv_img2lidar
        if stv_img_bytes:
            info['streetview_img_bytes'] = stv_img_bytes
        if stv_K:
            info['streetview_intrinsic'] = stv_K          # {cam: 3x3}
        if stv_E_cam2lidar:
            info['streetview_E_cam2lidar'] = stv_E_cam2lidar  # {cam: 4x4}
        if stv_fxfycxcy:
            info['streetview_fxfycxcy'] = stv_fxfycxcy    # {cam: [fx,fy,cx,cy]}

        lidar_sd_token = _get_lidar_sd_token_from_info(nusc, info)
        sat_entry = sat_db.get(lidar_sd_token, None)
        if sat_entry is None:
            miss_sat += 1
        else:
            sat_bytes_obj = sat_entry.get('img_bytes') or sat_entry.get('satellite_data') \
                            or sat_entry.get('img') or sat_entry.get('satellite_path')
            sat_bytes = _read_bytes_maybe(sat_bytes_obj, geoext_root)
            if sat_bytes is not None:
                info['satellite_img_bytes'] = sat_bytes
                try:
                    W, H = _size_from_bytes(sat_bytes)
                except Exception:
                    W = sat_entry.get('W'); H = sat_entry.get('H')
                    if W is None or H is None:
                        W, H = 400, 400
                info['satellite_pix2lidar'] = _pix2lidar_from_size((W, H), SAT_SCALE_M_PER_PX)

        if (idx + 1) % 500 == 0:
            print(f'  merged {idx+1}/{len(infos)}')

    print(f'[stat] miss_streetview={miss_stv}, miss_satellite={miss_sat}')

    out_path = bevdet_pkl_path.replace('.pkl', f'{out_suffix}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[save] {out_path}]')

if __name__ == '__main__':
    nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_DATAROOT, verbose=False)
    stv_db = _load_pkl(STREETVIEW_PKL)
    sat_db = _load_pkl(SATELLITE_PKL)

    GEOEXT_ROOT_HINT = None

    _merge_one(BEVDET_TRAIN_PKL, OUT_SUFFIX, nusc, stv_db, sat_db, GEOEXT_ROOT_HINT)
    _merge_one(BEVDET_VAL_PKL,   OUT_SUFFIX, nusc, stv_db, sat_db, GEOEXT_ROOT_HINT)
