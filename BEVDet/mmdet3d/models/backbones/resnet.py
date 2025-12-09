# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn
import torch

import os
import math
import torch.nn.functional as F
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck

from .MSDeformAttn import (
    MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16
)

def _select_msda_fn(dtype: torch.dtype):
    return (MultiScaleDeformableAttnFunction_fp16
            if dtype in (torch.float16, torch.bfloat16)
            else MultiScaleDeformableAttnFunction_fp32)

def _to_level_list(ctx):
    if ctx is None:
        return []
    if isinstance(ctx, (list, tuple)):
        lv = []
        for t in ctx:
            if t.ndim == 5:  # [B, Nc, C, H, W]
                lv.extend([t[:, i] for i in range(t.size(1))])
            elif t.ndim == 4:
                lv.append(t)
            else:
                raise ValueError(f"ref ndim={t.ndim}")
        return lv
    if ctx.ndim == 5:
        return [ctx[:, i] for i in range(ctx.size(1))]
    if ctx.ndim == 4:
        return [ctx]
    raise ValueError(f"ref ndim={ctx.ndim}")

def invert_img2lidar_to_lidar2img(img2lidar: torch.Tensor) -> torch.Tensor:

    assert img2lidar.ndim == 4 and img2lidar.shape[-2:] == (4, 4)
    orig_dtype = img2lidar.dtype
    M = torch.nan_to_num(img2lidar.to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)

    B, Nc = M.shape[:2]
    Mf = M.view(B * Nc, 4, 4)

    det = torch.det(Mf)
    good = det.abs() > 1e-6

    inv = torch.empty_like(Mf)
    if good.any():
        inv[good] = torch.linalg.inv(Mf[good])
    if (~good).any():
        inv[~good] = torch.linalg.pinv(Mf[~good], rcond=1e-6)

    return inv.view(B, Nc, 4, 4).to(orig_dtype)

@torch.no_grad()
def build_ref_cam_and_mask_from_img2lidar(
    reference_points: torch.Tensor,   # [B, num_query, D, 3], 
    pc_range: torch.Tensor,           # [6] = [xmin,ymin,zmin,xmax,ymax,zmax]
    img2lidar: torch.Tensor,          # [B, Nc, 4, 4]
    img_wh: torch.Tensor,             # [B, Nc, 2] = (W, H) 
):
    """
    返回：
      reference_points_cam: [Nc, B, num_query, D, 2]  (u,v) in [0,1]
      bev_mask            : [Nc, B, num_query, D]     bool
    """
    assert reference_points.ndim == 4 and reference_points.size(-1) == 3
    B, Q, D, _ = reference_points.shape
    _, Nc, _, _ = img2lidar.shape
    device, dtype = reference_points.device, reference_points.dtype

    lidar2img = invert_img2lidar_to_lidar2img(img2lidar)  # [B,Nc,4,4]

    pc_range = pc_range.to(device=device, dtype=dtype)
    xyz = reference_points.clone()
    xyz[..., 0] = xyz[..., 0] * (pc_range[3]-pc_range[0]) + pc_range[0]
    xyz[..., 1] = xyz[..., 1] * (pc_range[4]-pc_range[1]) + pc_range[1]
    xyz[..., 2] = xyz[..., 2] * (pc_range[5]-pc_range[2]) + pc_range[2]

    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [B,Q,D,4]
    xyz1 = xyz1.permute(2,0,1,3).contiguous()                       # [D,B,Q,4]
    xyz1 = xyz1.view(D, B, 1, Q, 4).expand(D, B, Nc, Q, 4)          # [D,B,Nc,Q,4]
    xyz1 = xyz1.unsqueeze(-1)                                       # [D,B,Nc,Q,4,1]

    P = lidar2img.to(torch.float32).view(1, B, Nc, 1, 4, 4).expand(D, B, Nc, Q, 4, 4)
    cam_pts = (P @ xyz1.to(torch.float32)).squeeze(-1)              # [D,B,Nc,Q,4]
    Z = cam_pts[..., 2:3]                                           # [D,B,Nc,Q,1]
    eps = 1e-5
    mask = (Z > eps)

    uv = cam_pts[..., 0:2] / torch.maximum(Z, torch.full_like(Z, eps))   # [D,B,Nc,Q,2]

    img_wh = img_wh.to(device=device, dtype=uv.dtype)
    WH = img_wh.view(1, B, Nc, 1, 2).expand(D, B, Nc, Q, 2)
    uv_norm = uv / WH  

    mask = (mask &
            (uv_norm[..., 0:1] > 0.0) & (uv_norm[..., 0:1] < 1.0) &
            (uv_norm[..., 1:2] > 0.0) & (uv_norm[..., 1:2] < 1.0))

    ref_cam = uv_norm.permute(2,1,3,0,4).contiguous()  # [Nc,B,Q,D,2]
    bev_mask = mask.squeeze(-1).permute(2,1,3,0).contiguous()  # [Nc,B,Q,D]

    return ref_cam.to(dtype), bev_mask


class ChannelLayerNorm(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)         # N H W C
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)      # N C H W


class AlphaBlender(nn.Module):
    def __init__(self, init_alpha=0.5, learned=True, use_gate=True, gate_init=-10.0):
        super().__init__()
        t = torch.tensor(init_alpha, dtype=torch.float32)
        self.alpha = nn.Parameter(t) if learned else nn.Parameter(t, requires_grad=False)
        self.gate  = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32)) if use_gate else None

    def forward(self, a, b, coef=None):
        B, dev, dt = a.shape[0], a.device, a.dtype
        shp = (B,) + (1,) * (a.ndim - 1)

        base = torch.sigmoid(self.alpha).to(device=dev, dtype=dt).view(1).expand(B).reshape(shp)
        if self.gate is not None:
            k = torch.sigmoid(self.gate).to(device=dev, dtype=dt).view(1).expand(B).reshape(shp)
        else:
            k = torch.ones(shp, device=dev, dtype=dt)

        # --- prepare coef ---
        if coef is None:
            c = torch.full((B,), 0.5, device=dev, dtype=dt)
        else:
            c = torch.as_tensor(coef, device=dev, dtype=dt).view(-1)
            if c.numel() == 1:
                c = c.expand(B)
            elif c.numel() != B:
                if B % c.numel() == 0:
                    c = c.repeat_interleave(B // c.numel())
                else:
                    raise ValueError(f"coef with {c.numel()} elements cannot broadcast to batch {B}")
        c = c.clamp(0, 1).reshape(shp)

        # coef 越大越偏 b
        alpha = (base + k * (c - base)).clamp(0, 1)
        return (1.0 - alpha) * a + alpha * b

class WindowCrossAttn2DRect(nn.Module):

    def __init__(self, dim_q, dim_ctx, num_heads=8, ws=None, stride=None, gsize=(4,4), zero_init_out=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_q, num_heads, batch_first=True, dropout=0.0)
        self.proj = nn.Conv2d(dim_q, dim_q, 1)
        if zero_init_out:
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x, ctx):
        """
        x:   [B, Cq, Hq, Wq]
        ctx: [B, Cc, Hk, Wk]
        """
        B, _, Hq, Wq = x.shape

        q_seq = x.flatten(2).transpose(1, 2).contiguous()   # [B, Sq, C]
        k_seq = ctx.flatten(2).transpose(1, 2).contiguous()   # [B, Sk, C]
        v_seq = ctx.flatten(2).transpose(1, 2).contiguous()   # [B, Sk, C]

        y_seq, _ = self.mha(q_seq, k_seq, v_seq)            # [B, Sq, C]

        C = y_seq.size(-1)
        y = y_seq.transpose(1, 2).contiguous().view(B, C, Hq, Wq)
        return self.proj(y)

class MSDeformRefAttn2D(nn.Module):

    def __init__(self, embed_dim, num_heads=8, num_points=4, im2col_step=64):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.num_points  = num_points
        self.im2col_step = im2col_step

        self.norm_in  = ChannelLayerNorm(embed_dim)
        self.ff_in    = nn.Conv2d(embed_dim, embed_dim, 1)
        self.norm_out = ChannelLayerNorm(embed_dim)
        self.ff_out   = nn.Conv2d(embed_dim, embed_dim, 1)

        self.offset_head = nn.Conv2d(embed_dim, num_heads * num_points * 2, 1)
        self.attn_head   = nn.Conv2d(embed_dim, num_heads * num_points,     1)

        with torch.no_grad():
            thetas = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
            grid = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid = grid / grid.abs().max(-1, keepdim=True)[0]
            grid = grid.view(num_heads, 1, 1, 2).repeat(1, num_points, 1, 1)
            for i in range(num_points):
                grid[:, i, 0, :] *= (i + 1)
            nn.init.zeros_(self.offset_head.weight)
            self.offset_head.bias.copy_(grid.view(-1))
            nn.init.zeros_(self.attn_head.weight)
            nn.init.zeros_(self.attn_head.bias)

    @staticmethod
    def _flatten_ref(ref):
        if ref is None:
            return []
        if isinstance(ref, (list, tuple)):
            outs = []
            for t in ref:
                assert t.ndim == 5, f"ref level must be [B,6,C,H,W], got {t.shape}"
                B, N, C, H, W = t.shape
                for i in range(N):
                    outs.append(t[:, i])  # [B,C,H,W]
            return outs
        else:
            assert ref.ndim == 5, f"ref must be [B,6,C,H,W], got {ref.shape}"
            B, N, C, H, W = ref.shape
            return [ref[:, i] for i in range(N)]

    @staticmethod
    def _build_ref_grid(Hq, Wq, device, dtype):
        ys = torch.linspace(0.5/Hq, 1 - 0.5/Hq, Hq, device=device, dtype=dtype)
        xs = torch.linspace(0.5/Wq, 1 - 0.5/Wq, Wq, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([xx, yy], dim=-1).view(1, Hq*Wq, 2)  # [1,HWq,2] in [0,1]
    
    def forward(self, x, ref, ref_img2lidar, img_wh, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        """
        x        : [B, Ce, Hq, Wq]  （BEV Query）
        ref      : [B, Nc, Ce, Hl, Wl] 或 list/tuple([B, Nc, Ce, Hl, Wl])
        ref_img2lidar : [B, Nc, 4, 4]
        img_wh   : [B, Nc, 2]  (W,H)
        pc_range : [6]
        """
        if ref is None:
            return x

        device, dtype = x.device, x.dtype
        x = x.contiguous()
        img_wh = img_wh.to(device=device, dtype=torch.float32).contiguous()
        ref_img2lidar = ref_img2lidar.to(device=device, dtype=torch.float32).contiguous()

        B, Ce, Hq, Wq = x.shape
        Q = Hq * Wq
        h = self.ff_in(self.norm_in(x)).contiguous()  # [B, Ce, Hq, Wq]

        levels = self._flatten_ref(ref)  # list of [B, Ce, Hl, Wl]
        levels = [t.contiguous() for t in levels]  

        Hl_list = [int(t.size(-2)) for t in levels]
        Wl_list = [int(t.size(-1)) for t in levels]
        shapes = torch.stack([
            torch.tensor([h_, w_], device=device, dtype=torch.long)
            for h_, w_ in zip(Hl_list, Wl_list)
        ], dim=0).contiguous()  # [L,2]

        areas = torch.tensor([h_*w_ for h_, w_ in zip(Hl_list, Wl_list)],
                            device=device, dtype=torch.long).contiguous()
        prefix = torch.zeros(1, device=device, dtype=torch.long)
        level_start_index = torch.cat([prefix, areas.cumsum(0)[:-1]], dim=0).contiguous()  # [L]

        head_dim = Ce // self.num_heads
        value = torch.cat([t.flatten(2).transpose(1, 2).contiguous() for t in levels], dim=1)
        value = value.view(B, -1, self.num_heads, head_dim).contiguous()

        off = torch.tanh(self.offset_head(h)).contiguous()                      # [B, H*K*2, Hq, Wq]
        off = off.view(B, self.num_heads, self.num_points, 2, Hq, Wq).contiguous()
        dx = off[:, :, :, 0].permute(0, 3, 4, 1, 2).contiguous().view(B, Q, self.num_heads, self.num_points)
        dy = off[:, :, :, 1].permute(0, 3, 4, 1, 2).contiguous().view(B, Q, self.num_heads, self.num_points)

        att = self.attn_head(h).view(B, self.num_heads, self.num_points, Hq, Wq).contiguous()
        att = F.softmax(att, dim=2)
        att = att.permute(0, 3, 4, 1, 2).contiguous().view(B, Q, self.num_heads, self.num_points)

        ref_grid = self._build_ref_grid(Hq, Wq, device, dtype).view(1, Hq, Wq, 2).contiguous()
        ref_grid = ref_grid.expand(B, Hq, Wq, 2).contiguous().view(B, Q, 2)  # [B,Q,2]
        z_anchor = torch.full((B, Q, 1), 0.5, device=device, dtype=dtype).contiguous()
        ref_bvz = torch.cat([ref_grid, z_anchor], dim=-1).unsqueeze(2).contiguous()  # [B,Q,1,3]

        ref_cam, bev_mask = build_ref_cam_and_mask_from_img2lidar(
            reference_points = ref_bvz,
            pc_range         = torch.as_tensor(pc_range, device=device, dtype=dtype),
            img2lidar        = ref_img2lidar,   # [B,Nc,4,4]
            img_wh           = img_wh,          # [B,Nc,2] (W,H)
        )
        # ref_cam: [Nc,B,Q,1,2]；bev_mask: [Nc,B,Q,1]
        Nc = int(ref_cam.shape[0])

        L = int(shapes.size(0))
        num_scales = L // Nc  

        sampling_list = []
        vismask_list = []
        idx_lv = 0
        for _s in range(num_scales):
            for cam in range(Nc):
                Hl = int(shapes[idx_lv, 0].item())
                Wl = int(shapes[idx_lv, 1].item())

                base = ref_cam[cam].permute(1, 2, 0, 3).contiguous().view(B, Q, 1, 2)  # [B,Q,1,2]
                loc_x = (base[..., 0:1] + dx / float(Wl)).contiguous()  # [B,Q,H,K]
                loc_y = (base[..., 1:2] + dy / float(Hl)).contiguous()
                sampling_list.append(torch.stack([loc_x, loc_y], dim=-1).contiguous())  # [B,Q,H,K,2]

                m = bev_mask[cam].permute(1, 2, 0).contiguous().view(B, Q, 1, 1)
                m = m.expand(B, Q, self.num_heads, self.num_points).contiguous()        # [B,Q,H,K]
                vismask_list.append(m)
                idx_lv += 1

        sampling_locations = torch.stack(sampling_list, dim=3).contiguous()  # [B,Q,H,L,K,2]
        vis_mask = torch.stack(vismask_list, dim=3).contiguous()             # [B,Q,H,L,K]

        att = att.unsqueeze(3).expand(B, Q, self.num_heads, L, self.num_points).contiguous()
        att = (att * vis_mask).contiguous()
        att = att / (att.sum(dim=(3, 4), keepdim=True) + 1e-6)

        msda = _select_msda_fn(dtype)
        out = msda.apply(
            value.contiguous(),
            shapes.contiguous(),
            level_start_index.contiguous(),
            sampling_locations.contiguous(),
            att.contiguous(),
            self.im2col_step,
        )  # [B, Q, Ce]
        out = out.transpose(1, 2).contiguous().view(B, Ce, Hq, Wq)

        y = self.ff_out(self.norm_out(out)).contiguous()
        return x + y


class CrossGateBlock2D(nn.Module):
    def __init__(self, dim, dim_ref=None, dim_bev=None,
                 heads_ref=8, heads_bev=8, learned_alpha=True,
                 ws=None, gsize=(4,4)):
        super().__init__()
        self.norm_in = ChannelLayerNorm(dim)

        self.ca_ref = None
        if dim_ref is not None:
            self.ca_ref = WindowCrossAttn2DRect(dim, dim_ref, heads_ref, ws=(8,22), stride=None, gsize=gsize)
        self.blend_ref = AlphaBlender(0.5, learned=learned_alpha)

        self.mid_norm = ChannelLayerNorm(dim)

        self.ca_bev = None
        if dim_bev is not None:
            self.ca_bev = WindowCrossAttn2DRect(dim, dim_bev, heads_bev, ws=(25,25), stride=None, gsize=gsize)
        self.blend_bev = AlphaBlender(0.5, learned=learned_alpha)

        self.norm_out = ChannelLayerNorm(dim); self.ff_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, ref=None, bev=None, coef=None, **_):
        B, C, H, W = x.shape
        h = self.norm_in(x)
        
        if self.ca_ref is not None and ref is not None:
            h = self.blend_ref(h, h + self.ca_ref(h, ref),coef)

        h = self.mid_norm(h)

        if self.ca_bev is not None and bev is not None:
            h = self.blend_bev(h, h + self.ca_bev(h, bev),coef)

        return x + self.ff_out(self.norm_out(h))



@BACKBONES.register_module()
class CustomResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class CustomResNetWithCross(CustomResNet):

    def __init__(self, *args, cross_ctor=None, cp_base=False, cp_cross=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_base  = cp_base
        self.cp_cross = cp_cross
        self._CP_KW = dict(use_reentrant=False)

        if len(args) > 0:
            numC_input = args[0]
        else:
            numC_input = kwargs['numC_input']
        num_layer = kwargs.get('num_layer', [2,2,2])
        num_channels = kwargs.get('num_channels', [numC_input*2**(i+1) for i in range(len(num_layer))])

        cross_stages = []
        for lvl, (stage, outC) in enumerate(zip(self.layers, num_channels)):
            xstage = []
            for bidx, _ in enumerate(stage):
                if cross_ctor is None:
                    cross = nn.Identity() 
                else:
                    mod = cross_ctor(lvl, bidx, outC)
                    cross = mod if mod is not None else nn.Identity()
                xstage.append(cross)
            cross_stages.append(nn.ModuleList(xstage))
        self.cross_stages = nn.ModuleList(cross_stages)

    def forward(self, x, ref_levels=None, bev_levels=None, coef=None):
        feats = []
        h = x
        for lvl, (stage, xstage) in enumerate(zip(self.layers, self.cross_stages)):
            ref = None if (ref_levels is None or ref_levels is False) else ref_levels[lvl]
            bev = None if (bev_levels is None or bev_levels is False) else bev_levels[lvl]
            for bidx, (base_block, cross_block) in enumerate(zip(stage, xstage)):
                if self.training and self.cp_base:
                    h = torch.utils.checkpoint.checkpoint(base_block, h, **self._CP_KW)
                else:
                    h = base_block(h)
                if self.training and self.cp_cross:
                    h = torch.utils.checkpoint.checkpoint(lambda t: cross_block(t, ref=ref, bev=bev), h, **self._CP_KW)
                else:
                    h = cross_block(h, ref=ref, bev=bev, coef=coef)
            if lvl in self.backbone_output_ids:
                feats.append(h)
        return feats
    
    
class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


@BACKBONES.register_module()
class CustomResNet3D(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(CustomResNet3D, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=3,
                        stride=stride[i],
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None))
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats