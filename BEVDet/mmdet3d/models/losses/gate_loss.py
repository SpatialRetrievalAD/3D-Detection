import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
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
