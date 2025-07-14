import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from misc.config import Params
from models.attention import SEBlock


def fasterrccn_mobile(params: Params):
    if not params.attention:
        return fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=2, weights_backbone=None)

    model = FasterRCNNWithAttention(num_classes=2)
    return model


class FasterRCNNWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=num_classes, weights_backbone=None)

        # Inject SE attention blocks after FPN outputs
        self.attention = nn.ModuleDict({
            name: SEBlock(ch.shape[1]) for name, ch in self.model.backbone(torch.randn(1, 3, 300, 300)).items()
        })

    def forward(self, images, targets=None):
        # Let the model handle transforms, backbone, etc.
        if self.training:
            loss_dict = self.model(images, targets)
            return loss_dict
        else:
            # Eval mode; apply attention manually after backbone if needed
            features = self.model.backbone(images.tensors if hasattr(images, "tensors") else images)
            features = {k: self.attention[k](v) for k, v in features.items()}
            proposals, _ = self.model.rpn(images, features)
            detections, _ = self.model.roi_heads(features, proposals, images.image_sizes)
            return self.model.transform.postprocess(detections, images.image_sizes)

