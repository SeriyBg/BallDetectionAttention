from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from misc.config import Params
from models.attention import SEBlock


def fasterrccn(params: Params):
    if not params.attention:
        return fasterrcnn_mobilenet_v3_large_fpn(Pretrained=False, num_classes=2, pretrained_backbone=False)

    backbone = mobilenet_v3_large(weights=None).features
    # Add SE block after block 12
    backbone[12] = nn.Sequential(
        backbone[12],
        SEBlock(channels=backbone[12][-1].out_channels)  # adjust based on final conv
    )
    return_layers = {'3': '0', '6': '1', '12': '2', '16': '3'}  # Customize as needed

    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=[40, 112, 160, 960],  # Must match chosen layers
        out_channels=256
    )

    # Final detector model
    model = FasterRCNN(backbone_with_fpn, num_classes=2)
    return model