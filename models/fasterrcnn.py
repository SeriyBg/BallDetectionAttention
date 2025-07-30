from torch import nn
from torchvision.models import mobilenet_v3_large, resnet50, MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2, \
    FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor

from misc.config import Params
from models.attention import attention_block


def fasterrccn(params: Params):
    if not params.attention:
        return fasterrcnn_resnet50_fpn_v2(num_classes=2, weight=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    model = fasterrcnn_resnet50_fpn_attention(num_classes=2, attention_type=params.attention_backbone_type)
    return model


def fasterrcnn_resnet50_fpn_attention(num_classes=2, attention_type = 'se'):
    backbone = resnet50(norm_layer=nn.BatchNorm2d, num_classes=num_classes, weights=ResNet50_Weights.IMAGENET1K_V1)

    # Insert SE blocks into selected layers of the backbone
    # We modify layers2 and layers3 (which feed into the FPN)
    def insert_se(module):
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if hasattr(block, 'conv3'):
                    out_channels = block.conv3.out_channels
                    block.add_module(attention_type, attention_block(attention_type, out_channels))
        return module

    backbone.layer2 = insert_se(backbone.layer2)
    backbone.layer3 = insert_se(backbone.layer3)

    # Wrap it with FPN
    backbone_with_fpn = _resnet_fpn_extractor(
        backbone=backbone,
        trainable_layers=5,
        norm_layer=nn.BatchNorm2d,
    )

    # Final detector
    model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
    model.backbone = backbone_with_fpn
    return model


def fasterrccn_mobilnet(params: Params):
    if not params.attention:
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn_attention(num_classes=2, attention_type=params.attention_backbone_type)
    return model


def fasterrcnn_mobilenet_v3_large_fpn_attention(num_classes, attention_type = 'se'):
        norm_layer = nn.BatchNorm2d

        trainable_backbone_layers = 4
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, norm_layer=norm_layer, num_classes=num_classes)
        attention_layers = [2, 4, 6, 9]
        for idx in attention_layers:
            layer = backbone.features[idx]
            if isinstance(layer, nn.Sequential) and hasattr(layer[0], 'out_channels'):
                out_ch = layer[0].out_channels
            elif hasattr(layer, 'out_channels'):
                out_ch = layer.out_channels
            else:
                continue  # skip if not a conv layer
            backbone.features[idx] = nn.Sequential(layer, attention_block(attention_type, out_ch))
        backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
        model.backbone = backbone
        return model
