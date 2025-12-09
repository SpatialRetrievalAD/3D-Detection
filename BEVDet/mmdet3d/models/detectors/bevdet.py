# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from mmcv.runner import force_fp32

import numpy as np
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet

from mmdet3d.models.backbones.resnet import CrossGateBlock2D, CustomResNetWithCross


import timm




###################### for ref #####################################


def zncc_diffmap_bn(x, y, k=9, eps=1e-5):
    """
    x, y: (B, N, C, H, W)
    return: diff (B, N, 1, H, W)  ∈[0,1]
    """
    B, N, C, H, W = x.shape
    if C != 1:
        x = x.mean(dim=2, keepdim=True)
        y = y.mean(dim=2, keepdim=True)

    x = x.view(B * N, 1, H, W)
    y = y.view(B * N, 1, H, W)
    pad = k // 2

    mx = F.avg_pool2d(x, k, 1, pad)
    my = F.avg_pool2d(y, k, 1, pad)
    vx = F.avg_pool2d((x - mx) ** 2, k, 1, pad)
    vy = F.avg_pool2d((y - my) ** 2, k, 1, pad)
    vxy = F.avg_pool2d((x - mx) * (y - my), k, 1, pad)

    zncc = vxy / (torch.sqrt(vx * vy) + eps)   # [-1, 1]
    diff = (1 - zncc) * 0.5                    # [0, 1]
    return diff.view(B, N, 1, H, W)


class CoefHeadRef(nn.Module):
    """
        diff: (B, N, 1, H, W)
        dist: (B, N) 
        coef: (B, N) ∈ (0,1)
    """
    def __init__(self, dist_scale=10.0):
        super().__init__()
        self.dist_scale = dist_scale
        self.img_head = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1)
        )
        self.dist_head = nn.Linear(1, 1)

    def forward(self, diff, dist=None):
        B, N, _, H, W = diff.shape
        x = diff.view(B * N, 1, H, W)
        img_logit = self.img_head(x).view(B, N)  # (B,N)

        if dist is not None:
            d01 = torch.tanh(dist / self.dist_scale)        # (B,N)
            dist_logit = self.dist_head(d01.unsqueeze(-1)).squeeze(-1)  # (B,N)
        else:
            dist_logit = 0.0

        logit = img_logit + dist_logit    # [B, N]
        coef = torch.sigmoid(logit)  # (B,N)
        coef = coef.mean(dim=1)      # [B]
        return coef  # [B] ∈ (0,1)

class _Linear1x1Stack(nn.Module):

    def __init__(self, in_ch: int, out_chs=(128, 256, 512)):
        super().__init__()
        self.out_chs = tuple(out_chs)
        self.projs = nn.ModuleList([nn.Conv2d(in_ch, c, 1, bias=True)
                                    for c in self.out_chs])
        for m in self.projs:
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):

        raise NotImplementedError

class RefPyramid1x1(_Linear1x1Stack):
    
    def __init__(self, in_ch: int, out_chs=(128, 256, 512), depthwise: bool = False):
        super().__init__(in_ch=in_ch, out_chs=out_chs)

        groups = in_ch if depthwise else 1
        self.mid_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1,
                                bias=False, groups=groups)
        nn.init.kaiming_normal_(self.mid_conv.weight, nonlinearity='relu')
        self.mid_act = nn.ReLU(inplace=True)

    def forward(self, ref_feat: torch.Tensor):
        """
        Args:
            ref_feat: Tensor[B, N, Cin, H, W]
        Returns:
            tuple of tensors, [B, Cout_i, 8*N, 22]
        """
        assert ref_feat.dim() == 5, "ref_feat must be [B,N,Cin,H,W]"
        B, N, Cin, H0, W0 = ref_feat.shape
        x = ref_feat.view(B * N, Cin, H0, W0).contiguous()
        
        # 2) 中间 3×3 卷积 stride2 + ReLU
        x = self.mid_conv(x)
        x = self.mid_act(x)
        target_size = x.shape[-2:]
        # 3) 多分支 1×1 线性投影；把 [B*N, Cout, 8, 22] -> (B, Cout, 8*N, 22)
        outs = []
        for p in self.projs:
            yi = p(x).contiguous()                         # [B*N, Cout, 8, 22]
            Cout = yi.size(1)
            yi = yi.view(B, N, Cout, *target_size)         # [B, N, Cout, 8, 22]
            yi = yi.permute(0, 2, 3, 1, 4).contiguous()    # [B, Cout, 8, N, 22]
            yi = yi.view(B, Cout, target_size[0]*N, target_size[1])    # [B, Cout, 8*N, 22]
            outs.append(yi)
        return tuple(outs)
        
class BevPyramid1x1(_Linear1x1Stack):
    def __init__(self, in_ch, out_chs=(128,256,512),
                 use_mid=True, depthwise=False, act='relu'):
        super().__init__(in_ch, out_chs)
        self.use_mid = use_mid
        if use_mid:
            groups = in_ch if depthwise else 1
            self.mid = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False, groups=groups)
            nn.init.kaiming_normal_(self.mid.weight, nonlinearity='relu')
            self.act = nn.ReLU(inplace=True) if act=='relu' else nn.SiLU(inplace=True)

    def forward(self, bev_feat: torch.Tensor):
        x = bev_feat.contiguous()
        if self.use_mid:
            x = self.act(self.mid(x))
        return tuple(p(x) for p in self.projs)
    

def make_cross_ctor(
    use_ref=True, use_bev=True,
    heads_ref=8, heads_bev=8,
    learned_alpha=True,
    ws_map=None,     # dict[level]->(wh, ww) 或 None=auto
    gsize=(4,4),
):

    def ctor(level, block_idx, outC):
        ws = None if (ws_map is None or level not in ws_map) else ws_map[level]
        return CrossGateBlock2D(
            dim=outC,
            dim_ref=(outC if use_ref else None),
            dim_bev=(outC if use_bev else None),
            heads_ref=heads_ref, heads_bev=heads_bev,
            learned_alpha=learned_alpha,
            ws=ws, gsize=gsize
        )
    return ctor

def set_trainable(module, flag: bool = True):
    for p in module.parameters():
        p.requires_grad = flag

###################### for 3dpe #####################################

def add_plucker_to_ref_feat(ref_feat, ray_o, ray_d, method="default_plucker"):
    """
    ref_feat: [B, V, C, Hf, Wf]
    ray_o, ray_d: [B, V, 3, Hr, Wr] 
    return: [B, V, C+K, Hf, Wf]，K=6(default/custom_plucker) 或 9(aug_plucker)
    """
    B, V, C, Hf, Wf = ref_feat.shape
    _, _, _, Hr, Wr = ray_o.shape

    if (Hr != Hf) or (Wr != Wf):
        ro = F.interpolate(ray_o.reshape(B*V, 3, Hr, Wr), size=(Hf, Wf),
                           mode='bilinear', align_corners=False)
        rd = F.interpolate(ray_d.reshape(B*V, 3, Hr, Wr), size=(Hf, Wf),
                           mode='bilinear', align_corners=False)
        ray_o = ro.reshape(B, V, 3, Hf, Wf)
        ray_d = rd.reshape(B, V, 3, Hf, Wf)

    ray_o = ray_o.to(ref_feat.dtype).to(ref_feat.device)
    ray_d = ray_d.to(ref_feat.dtype).to(ref_feat.device)

    o_cross_d = torch.cross(ray_o, ray_d, dim=2)                  # [B,V,3,Hf,Wf]
    pose_cond = torch.cat([o_cross_d, ray_d], dim=2)              # [B,V,6,Hf,Wf]

    return torch.cat([ref_feat, pose_cond], dim=2)

###################### for 3dpe #####################################



@DETECTORS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 gate_loss=None,
                 use_grid_mask=False,
                 use_bev=False,
                 use_ref=False,
                 **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)

        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.gate_loss = builder.build_loss(gate_loss)
        
        #### for ggearth #########################################################
        cross_ctor = make_cross_ctor(
                use_ref=True, use_bev=True,
                heads_ref=8, heads_bev=8,
                ws_map=None,
            )
        
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = CustomResNetWithCross(
            numC_input=64,
            num_layer=[2,2,2],
            stride=[2,2,2],
            block_type='Basic',
            cross_ctor=cross_ctor,
            cp_base=False,  
            cp_cross=False,
        )
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
            
            
        self.raw_backbone = timm.create_model(
            'resnet50', pretrained=False, features_only=True, out_indices=(3,)
        )

        ckpt = torch.load('xxx/ckpt/resnet50-0676ba61.pth',
                        map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}  
        model_sd = self.raw_backbone.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        self.raw_backbone.load_state_dict(filtered, strict=False)
        
        self.use_bev = use_bev
        self.use_ref = use_ref
    
        self.ref_pyr = RefPyramid1x1(in_ch=1030, out_chs=(128,256,512))
        self.bev_pyr = BevPyramid1x1(in_ch=1024, out_chs=(128,256,512))
        self.coefhead = CoefHeadRef()
        #### for ggearth #########################################################

        
        for p in self.parameters():
            p.requires_grad = False

        set_trainable(self.ref_pyr)

        set_trainable(self.bev_pyr)

        set_trainable(self.img_bev_encoder_backbone.cross_stages)
        
        set_trainable(self.pts_bbox_head)
        
        set_trainable(self.coefhead)

    
    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def _get_uv_grid(self, H, W, pad_h, pad_w, device, dtype, branch: str):
        if branch == 'ref':
            if self._pe_ref_cached_hw != (H, W):
                u = torch.arange(W, device=device, dtype=dtype) * (pad_w / W)
                v = torch.arange(H, device=device, dtype=dtype) * (pad_h / H)
                self._pe_ref_u, self._pe_ref_v = u, v
                self._pe_ref_cached_hw = (H, W)
            return self._pe_ref_u, self._pe_ref_v
        elif branch == 'cur':
            if self._pe_cur_cached_hw != (H, W):
                u = torch.arange(W, device=device, dtype=dtype) * (pad_w / W)
                v = torch.arange(H, device=device, dtype=dtype) * (pad_h / H)
                self._pe_cur_u, self._pe_cur_v = u, v
                self._pe_cur_cached_hw = (H, W)
            return self._pe_cur_u, self._pe_cur_v
        else:
            raise ValueError(f"unknown branch {branch}")
        
    @force_fp32()
    def bev_encoder(self, x, ref_levels, bev_levels, coef):
        x = self.img_bev_encoder_backbone(x, ref_levels, bev_levels, coef)        
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def prepare_ref_LSS(self, ego2globals, streetview_img2lidar, streetview_imgs, ref_intrinsic, **kwargs):
        # split the inputs into each frame
        B, N, C, H, W = streetview_imgs.shape
            
        imgs = streetview_imgs
        sensor2egos = streetview_img2lidar
        intrins = ref_intrinsic
        post_rots  = torch.eye(3).view(1,1,3,3).repeat(B,N,1,1).to(streetview_img2lidar.device)  # = I3
        post_trans = torch.zeros(B, N, 3).to(streetview_img2lidar.device)          # = 0
        bda        = torch.eye(4).view(1,4,4).repeat(B,1,1).to(streetview_img2lidar.device)       # = I4 per-sample
        
        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]
    
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        sensor2ego = img[1]
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        bs = x.shape[0]

        if not self.training:
            kwargs['streetview_imgs'] = kwargs['streetview_imgs'][0]
            kwargs['ref_ray_o'] = kwargs['ref_ray_o'][0]
            kwargs['ref_ray_d'] = kwargs['ref_ray_d'][0]
            kwargs['satellite_img'] = kwargs['satellite_img'][0]
            kwargs['ref_dist'] = kwargs['ref_dist'][0]
            
        B, N, C, H, W = kwargs['streetview_imgs'].shape
        diff = zncc_diffmap_bn(kwargs['streetview_imgs'][:, 1:2, ...],img[0][:, 1:2, ...])
        ref_feat = self.raw_backbone(kwargs['streetview_imgs'].contiguous().view(B * N, C, H, W))[0]
        BNe, C_out, H_out, W_out = ref_feat.shape
        ref_feat = ref_feat.view(B, N, C_out, H_out, W_out)
        ref_feat = add_plucker_to_ref_feat(
                ref_feat,                      # [B,6,1024,16,44]
                kwargs['ref_ray_o'],           # [B,6,3,256,704]
                kwargs['ref_ray_d'],           # [B,6,3,256,704]
                method="default_plucker"       # 或 "custom_plucker"/"aug_plucker"
            )
        dist = kwargs['ref_dist']
        coef = self.coefhead(diff, dist)

        B, C, H, W = kwargs['satellite_img'].shape
        bev_feat = self.raw_backbone(kwargs['satellite_img'])[0]
        
        ############################ end ref ####################################################

        x, depth = self.img_view_transformer([x] + img[1:7])
        ref_levels = self.ref_pyr(ref_feat) if self.use_ref else None
        bev_levels = self.bev_pyr(bev_feat) if self.use_bev else None
        
        x = self.bev_encoder(x, ref_levels, bev_levels, coef)
        return [x], depth, coef

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, coef = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, coef)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _, coef = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        if isinstance(kwargs['ref_quality'], list):
            streetview_quality_label = kwargs['ref_quality'][0]
        else:
            streetview_quality_label = kwargs['ref_quality']
            
        loss_streetview_quality = self.gate_loss(
            coef, streetview_quality_label.squeeze())
        losses.update(loss_streetview_quality)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False
        
    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVDepth4D(BEVDet4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVStereo4D(BEVDepth4D):
    def __init__(self, **kwargs):
        super(BEVStereo4D, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame