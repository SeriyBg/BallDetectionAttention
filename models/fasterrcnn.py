import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from misc.config import Params
from models.attention import SEBlock


def fasterrccn_mobile(params: Params):
    if not params.attention:
        return fasterrcnn_mobilenet_v3_large_fpn(Pretrained=False, num_classes=2, pretrained_backbone=False)

    model = FasterRCNNWithAttention(num_classes=2)
    return model


class FasterRCNNWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=False,
            pretrained_backbone=False,
            num_classes=num_classes
        )

        # Wrap attention on FPN outputs
        self.attention_blocks = nn.ModuleDict({
            k: SEBlock(v.shape[1]) for k, v in self.model.backbone.body(torch.randn(1, 3, 300, 300)).items()
        })

    def forward(self, images, targets=None):
        # Hook into the FPN backbone
        features = self.model.backbone(images.tensors if hasattr(images, "tensors") else images)

        # Apply attention to each feature map
        attended = {k: self.attention_blocks[k](v) for k, v in features.items()}

        # Continue as usual
        if self.training:
            return self.model.rpn(images, attended, targets)
        else:
            return self.model.roi_heads(self.model.rpn(images, attended)[0], attended, images.image_sizes)
