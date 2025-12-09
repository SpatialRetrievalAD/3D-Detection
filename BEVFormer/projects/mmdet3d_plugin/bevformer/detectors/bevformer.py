# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import timm
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import torch.nn.functional as F
import torch.nn as nn


class StreetviewQualityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_sigmoid=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()

    def forward(self, streetview_quality, streetview_quality_label):
        valid_mask = streetview_quality_label != -1
        if not valid_mask.any():
            return {'loss_streetview_quality': streetview_quality.sum() * 0}

        pred = streetview_quality[valid_mask].float()
        target = streetview_quality_label[valid_mask].float()

        loss = self.bce(pred, target)
        return {'loss_streetview_quality': self.loss_weight * loss}


def resize_keep_ratio_center_crop_bn(ref, target_h, target_w):
    """
    ref: (B, N, C, H, W)  ->  (B, N, C, target_h, target_w)
    """
    B, N, C, H, W = ref.shape
    ref_flat = ref.view(B * N, C, H, W)

    scale = max(target_h / H, target_w / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    ref_resized = F.interpolate(ref_flat, size=(new_h, new_w),
                                mode='bilinear', align_corners=False)
    crop_h = (new_h - target_h) // 2
    crop_w = (new_w - target_w) // 2
    ref_cropped = ref_resized[:, :, crop_h:crop_h+target_h, crop_w:crop_w+target_w]
    return ref_cropped.view(B, N, C, target_h, target_w)


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
        dist: (B, N) 或 None

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


def set_trainable(module, flag: bool = True):
    for p in module.parameters():
        p.requires_grad = flag

def resize_pose_curr(pose_curr, out_hw=(4, 7), mode="avg"):
    # pose_curr: [B, N, C=6, H, W] -> [B, N, C, 4, 7]
    B, N, C, H, W = pose_curr.shape
    x = pose_curr.view(B*N, C, H, W)
    x = F.adaptive_avg_pool2d(x, out_hw) if mode=="avg" else F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
    return x.view(B, N, C, *out_hw)

def resize_pose_prev(pose_prev, out_hw=(4, 7), mode="avg", merge_time=True):
    # pose_prev: [B, T, N, C=6, H, W]
    B, T, N, C, H, W = pose_prev.shape
    x = pose_prev.view(B*T*N, C, H, W)
    x = F.adaptive_avg_pool2d(x, out_hw) if mode=="avg" else F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
    x = x.view(B, T, N, C, *out_hw)
    return x.view(B, T*N, C, *out_hw) if merge_time else x

def build_pose_cond_default(prev_o, prev_d, curr_o, curr_d):
    pose_prev = torch.cat([torch.cross(prev_o, prev_d, dim=3), prev_d], dim=3)  # [B,T,N,6,H,W]
    pose_curr = torch.cat([torch.cross(curr_o, curr_d, dim=2), curr_d], dim=2)  # [B,N,6,H,W]
    return pose_prev, pose_curr

def curr_pose_cond_default(curr_o, curr_d):
    pose_curr = torch.cat([torch.cross(curr_o, curr_d, dim=2), curr_d], dim=2)  # [B,N,6,H,W]
    return pose_curr


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 gate_loss=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        
        # for ref ########################################################################
        
        self.gate_loss = StreetviewQualityLoss()
        self.coefhead = CoefHeadRef()
        for p in self.parameters():
            p.requires_grad = False

        set_trainable(self.pts_bbox_head.transformer.encoder.bev_fusion)
        set_trainable(self.pts_bbox_head.transformer.decoder)
        set_trainable(self.pts_bbox_head.cls_branches)
        set_trainable(self.pts_bbox_head.reg_branches)
        set_trainable(self.coefhead)

    def extract_street_feat(self, street_img, len_queue=None):
        # Skip if streetview is disabled or input is None

        B = street_img.size(0)
        if street_img is not None:

            if street_img.dim() == 5 and street_img.size(0) == 1:
                street_img.squeeze_()
            elif street_img.dim() == 5 and street_img.size(0) > 1:
                B, N, C, H, W = street_img.size()
                street_img = street_img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                street_img = self.grid_mask(street_img)
            
            img_feats = self.img_backbone(street_img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped       

    def extract_sat_feat(self, sat_img, len_queue=None):
        # Skip if satellite is disabled or input is None
        B = sat_img.size(0)
        assert sat_img.dim() in (4, 5), 'sat_img must be 4D/5D tensor'

        if sat_img.dim() == 5:
            if len_queue is not None:
                B, T, C, H, W = sat_img.shape
                x = sat_img.reshape(B * T, C, H, W)
            else:
                B, N1, C, H, W = sat_img.shape
                assert N1 == 1, 'satellite expects single-view per sample; got N={}'.format(N1)
                x = sat_img.view(B * N1, C, H, W)
        else:
            x = sat_img

        feats = self.img_backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        if self.with_img_neck:
            feats = self.img_neck(feats)
            
        sat_feats_reshaped = []
        for feat in feats:
            BL, C, H, W = feat.shape
            if len_queue is not None:
                sat_feats_reshaped.append(feat.view(int(B/len_queue), len_queue, C, H, W))
            else:
                sat_feats_reshaped.append(feat)
        
        return sat_feats_reshaped

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,ref_feats=None,
                          bev_feats=None,coef=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, ref_feats=ref_feats, bev_feats=bev_feats,coef=coef)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list, prev_streetview=None, 
                           prev_satellite=None, pose_prev=None, coef=None):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            if prev_streetview is not None:
                bs_s, len_q_s, num_cams_s, C_s, H_s, W_s = prev_streetview.shape
                prev_streetview = prev_streetview.reshape(bs_s*len_q_s, num_cams_s, C_s, H_s, W_s)
                street_feats_list = self.extract_street_feat(prev_streetview, len_queue=len_queue)
                pose_prev = pose_prev.reshape(bs,len_queue,num_cams,6, 4, 7)
                
                
            if prev_satellite is not None:
                bs_sat, len_q_sat, C_sat, H_sat, W_sat = prev_satellite.shape
                prev_satellite = prev_satellite.reshape(bs_sat*len_q_sat, C_sat, H_sat, W_sat)
                sat_feats_list = self.extract_sat_feat(prev_satellite, len_queue=len_queue)
            
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                    
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                pre_ref_feats = [each_scale[:, i] for each_scale in street_feats_list][-1]
                pre_bev_feats = [each_scale[:, i] for each_scale in sat_feats_list][-1]
                
                pre_ref_feats=  torch.cat([pre_ref_feats, pose_prev[:, i]], dim=2)
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True, 
                    ref_feats=pre_ref_feats, bev_feats=pre_bev_feats, coef=coef)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      **kwargs,
                      ):
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
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        B, N, C, H, W = img.shape

        prev_streetview = kwargs['streetview_imgs'][:, :-1, ...] if len_queue > 1 else None
        streetview = kwargs['streetview_imgs'][:, -1, ...]
        ref_feat = self.extract_street_feat(streetview)[-1]
    
        prev_ref_ray_o = kwargs['ref_ray_o'][:, :-1, ...] if len_queue > 1 else None
        ref_ray_o = kwargs['ref_ray_o'][:, -1, ...]
        prev_ref_ray_d = kwargs['ref_ray_d'][:, :-1, ...] if len_queue > 1 else None
        ref_ray_d = kwargs['ref_ray_d'][:, -1, ...]
        pose_prev, pose_curr = build_pose_cond_default(prev_ref_ray_o, prev_ref_ray_d, ref_ray_o, ref_ray_d)
        pose_curr = resize_pose_curr(pose_curr, (4,7), mode="avg")      # [B, N, 6, 4, 7]
        pose_prev = resize_pose_prev(pose_prev, (4,7), mode="avg", merge_time=True)  # [B, T*N, 6, 4, 7]
        ref_feat = torch.cat([ref_feat, pose_curr], dim=2)
        
        prev_satellite = kwargs['satellite_img'][:, :-1, ...] if len_queue > 1 else None
        satellite = kwargs['satellite_img'][:, -1, ...]
        bev_feat = self.extract_sat_feat(satellite)[-1]

        ref_resized = resize_keep_ratio_center_crop_bn(kwargs['streetview_imgs'][:, -1, ...][:, 1:2, ...], H, W)
        diff = zncc_diffmap_bn(ref_resized,img[:, 1:2, ...])
        dist = kwargs['ref_dist']
        coef = self.coefhead(diff, dist)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, prev_streetview, prev_satellite, pose_prev, coef)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev,
                                            ref_feat,bev_feat, coef)
        loss_streetview_quality = self.gate_loss(
            coef, kwargs['ref_quality'].squeeze())
        losses.update(loss_streetview_quality)
        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, ref_feat=None, bev_feat=None):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, ref_feats=ref_feat, bev_feats=bev_feat)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        streetview = kwargs['streetview_imgs'][0]
        ref_feat = self.extract_street_feat(streetview)[-1]
        
        ref_ray_o = kwargs['ref_ray_o'][0]
        ref_ray_d = kwargs['ref_ray_d'][0]
        pose_curr = curr_pose_cond_default(ref_ray_o, ref_ray_d)
        pose_curr = resize_pose_curr(pose_curr, (4,7), mode="avg")      # [B, N, 6, 4, 7]
        ref_feat = torch.cat([ref_feat, pose_curr], dim=2)

        satellite = kwargs['satellite_img'][0]
        bev_feat = self.extract_sat_feat(satellite)[-1]
        
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale, ref_feat=ref_feat, bev_feat=bev_feat)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
