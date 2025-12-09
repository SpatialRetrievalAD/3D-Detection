
# Copyright (c) OpenMMLab. All rights reserved.
import os
import io

import cv2
import mmcv
import json
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from einops import rearrange
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.datasets.builder import PIPELINES



def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean=np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    to_bgr = True
    img = imnormalize(np.array(img), mean, std, to_bgr)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class LoadRefs(object):
    def __init__(self, data_config, quality_json):
        self.data_config = data_config
        self.cams_order = data_config['cams']  
        self.num_cams = len(self.cams_order)
        
        K = torch.tensor([
            [553.4595,   0.0000, 352.0000],
            [  0.0000, 553.4595, 128.0000],
            [  0.0000,   0.0000,   1.0000]
        ], dtype=torch.float32)

        self.refK_expand = K.unsqueeze(0).expand(self.num_cams, -1, -1)  # [6,3,3], view
        
        self.p_drop_sv  = float(data_config.get('p_drop_streetview', 0.0))
        self.p_drop_sat = float(data_config.get('p_drop_satellite', 0.0))
        self.drop_sv_per_cam = bool(data_config.get('drop_sv_per_cam', False))
        
        self.quality_map = None
        if quality_json and os.path.isfile(quality_json):
            with open(quality_json, "r", encoding="utf-8") as f:
                qraw = json.load(f)
            self.quality_map = {}
            for k, v in qraw.items():
                val = v.get("label", -1) if isinstance(v, dict) else v
                try: val = int(val)                     
                except (TypeError, ValueError): val = -1
                if val not in (0, 1): val = -1           # 只认0/1，其它当缺失
                self.quality_map[str(k)] = val           # 统一成字符串key
    
    @staticmethod
    def compute_rays(c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        c2w: [b, v, 4, 4] 
        fxfycxcy: [b, v, 4]
        ray_o, ray_d : [b, v, 3, h, w]
        """
        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:, :, 0], fxfycxcy[:, :, 1], fxfycxcy[:, :, 2], fxfycxcy[:, :, 3]
        eps = torch.finfo(fx.dtype).eps
        fx = fx.clamp_min(1e-6); fy = fy.clamp_min(1e-6)

        h_orig = int((2 * cy.max()).clamp_min(2).item())
        w_orig = int((2 * cx.max()).clamp_min(2).item())
        if h is None or w is None:
            h, w = h_orig, w_orig

        if h_orig != h or w_orig != w:
            fx = fx * w / max(w_orig, 1)
            fy = fy * h / max(h_orig, 1)
            cx = cx * w / max(w_orig, 1)
            cy = cy * h / max(h_orig, 1)

        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).reshape(b * v, 4)

        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)

        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        R = c2w[:, :3, :3].transpose(1, 2)
        ray_d = torch.bmm(ray_d, R)
        ray_d = ray_d / (ray_d.norm(dim=2, keepdim=True).clamp_min(1e-6))

        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        return ray_o, ray_d

    def __call__(self, results):
        curr = results
        fH, fW = tuple(self.data_config['input_size'])
        N = len(self.cams_order)

        sv_bytes = curr.get('streetview_img_bytes', None)  # dict 或 None
        sv_pil = {}
        if sv_bytes is not None:
            sv_tensors = []
            for cam in self.cams_order:
                b = sv_bytes[cam]
                img = Image.open(io.BytesIO(b)).convert('RGB')
                sv_pil[cam] = img
                sv_tensors.append(mmlabNormalize(img))
            streetview_tensors = torch.stack(sv_tensors, dim=0)  # [6,3,fH,fW]
        else:
            streetview_tensors = torch.zeros((N, 3, fH, fW), dtype=torch.float32)
            sv_pil = {cam: None for cam in self.cams_order}

        if torch.rand(()) < self.p_drop_sv:
            streetview_tensors = torch.zeros_like(streetview_tensors)
        
        results['streetview_imgs'] = streetview_tensors

        stv_i2l = curr.get('streetview_img2lidar', None)  # dict 或 None
        if stv_i2l is not None:
            mats = []
            for cam in self.cams_order:
                M = stv_i2l[cam]                     # 4x4
                mats.append(torch.as_tensor(M, dtype=torch.float32))
            results['streetview_img2lidar'] = torch.stack(mats, dim=0)  # [6,4,4]
        else:
            results['streetview_img2lidar'] = torch.zeros((N, 4, 4), dtype=torch.float32)

        sat_p2l = curr.get('satellite_pix2lidar', None)
        if sat_p2l is not None:
            results['satellite_pix2lidar'] = torch.as_tensor(sat_p2l, dtype=torch.float32).reshape(3, 3)
        else:
            results['satellite_pix2lidar'] = torch.zeros((3, 3), dtype=torch.float32)

        sat_bytes = curr.get('satellite_img_bytes', None)
        if sat_bytes is not None:
            sat_img = Image.open(io.BytesIO(sat_bytes)).convert('RGB')   
            satellite_tensor = mmlabNormalize(sat_img)           # [3,fH,fW]
            if self.p_drop_sat > 0.0 and torch.rand(()) < self.p_drop_sat:
                satellite_tensor = torch.zeros_like(satellite_tensor)
            
            results['satellite_img'] = satellite_tensor
        else:
            results['satellite_img'] = torch.zeros((3, 400, 400), dtype=torch.float32)
            
        stv_intrinsic = curr.get('streetview_intrinsic', None)  # {cam: 3x3} 或 None
        if stv_intrinsic is not None:
            Ks = []
            for cam in self.cams_order:
                Ks.append(torch.as_tensor(stv_intrinsic[cam], dtype=torch.float32))
            K_stack = torch.stack(Ks, dim=0)  # [6,3,3]
        else:
            K_stack = self.refK_expand  # [6,3,3]
        results['ref_intrinsic'] = K_stack

        fx = K_stack[:, 0, 0]
        fy = K_stack[:, 1, 1]
        cx = K_stack[:, 0, 2]
        cy = K_stack[:, 1, 2]
        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)  # [6,4]
        results['fxfycxcy'] = fxfycxcy

        stv_extr = curr.get('streetview_E_cam2lidar', None)  # {cam: 4x4} 或 None
        if stv_extr is not None:
            mats = [torch.as_tensor(stv_extr[cam], dtype=torch.float32) for cam in self.cams_order]
            c2w = torch.stack(mats, dim=0)  # [6,4,4]
        else:
            c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1).clone()  # [6,4,4]
        results['streetview_E_cam2lidar'] = c2w
        c2w_b = c2w.unsqueeze(0)                 # [1,6,4,4]
        fxfycxcy_b = fxfycxcy.unsqueeze(0)       # [1,6,4]
        device = streetview_tensors.device

        ray_o, ray_d = self.compute_rays(c2w_b, fxfycxcy_b, h=fH, w=fW, device=device)
        results['ref_ray_o'] = ray_o.squeeze(0)             # [6,3,fH,fW]
        results['ref_ray_d'] = ray_d.squeeze(0)             # [6,3,fH,fW]
        
        # ------------ ref_quality -------------
        sample_token = curr.get("token", None)
        if sample_token is not None and getattr(self, "quality_map", None) is not None:
            try:
                q = int(self.quality_map.get(str(sample_token), -1))
                q = q if q in (0, 1) else -1
            except Exception:
                q = -1
        else:
            q = -1
        results['ref_quality'] = torch.tensor([[q]], dtype=torch.int8)

        # ------------ ref_dist ----------------
        if stv_i2l is not None:
            i2l = results['streetview_img2lidar']   # [6,4,4]
            t0 = i2l[0, :3, 3]                      # cam0 的平移
            dist = torch.linalg.norm(t0).float()
        else:
            dist = torch.tensor(1000.0, dtype=torch.float32)
        results['ref_dist'] = dist

        return results
