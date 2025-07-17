from torch import nn
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _validate_trainable_layers

from misc.config import Params
from models.attention import SEBlock


def fasterrccn(params: Params):
    if not params.attention:
        return fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2, weights_backbone=None)

    model = fasterrcnn_resnet50_fpn_attention(num_classes=2)
    return model


def fasterrcnn_resnet50_fpn_attention(num_classes=2):
    model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2, weights_backbone=None)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)

    # Add attention to specific ResNet layers
    backbone = model.backbone.body

    backbone.layer2[1].relu = nn.Sequential(backbone.layer2[1].relu, SEBlock(512))
    backbone.layer3[1].relu = nn.Sequential(backbone.layer3[1].relu, SEBlock(1024))
    backbone.layer4[1].relu = nn.Sequential(backbone.layer4[1].relu, SEBlock(2048))

    return model


def fasterrccn_mobilnet(params: Params):
    if not params.attention:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=2, weights_backbone=None)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn_attention(num_classes=2)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    return model


def fasterrcnn_mobilenet_v3_large_fpn_attention(num_classes):
        norm_layer = nn.BatchNorm2d

        trainable_backbone_layers = _validate_trainable_layers(False, None, 6, 3)
        trainable_backbone_layers = 4
        backbone = mobilenet_v3_large(weights=None, norm_layer=norm_layer)
        se_layers = [2, 4, 6, 9]
        for idx in se_layers:
            layer = backbone.features[idx]
            if isinstance(layer, nn.Sequential) and hasattr(layer[0], 'out_channels'):
                out_ch = layer[0].out_channels
            elif hasattr(layer, 'out_channels'):
                out_ch = layer.out_channels
            else:
                continue  # skip if not a conv layer
            backbone.features[idx] = nn.Sequential(layer, SEBlock(out_ch))
        backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
        anchor_sizes = (
                           (
                               32,
                               64,
                               128,
                               256,
                               512,
                           ),
                       ) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model = FasterRCNN(
            backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios)
        )

        return model
