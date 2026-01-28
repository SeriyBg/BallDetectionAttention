from torch import nn
from torchvision.models.detection import ssd300_vgg16

from models.attention import attention_block, ApplyAttentionToList


def ssd_ball_detector(attention: bool = False, attention_backbone_type = None, attention_head_type = None, attention_transition_type=None):
    # Create SSD model with 2 classes (background + ball)
    num_classes = 2  # background + ball
    if attention:
        return ssd_ball_detector_attention(num_classes, attention_backbone_type, attention_head_type, attention_transition_type)
    return ssd300_vgg16(num_classes=num_classes)

def ssd_ball_detector_attention(num_classes, attention_backbone_type, attention_head_type, attention_transition_type):
    model = ssd300_vgg16(num_classes=num_classes)

    if attention_backbone_type is not None:
        vgg_features = model.backbone.features
        vgg_features[15] = nn.Sequential(vgg_features[15], attention_block(attention_backbone_type, 256))  # after conv3_3
        vgg_features[22] = nn.Sequential(vgg_features[22], attention_block(attention_backbone_type, 512))  # after conv4_3
        model.backbone.features = vgg_features

    if attention_transition_type is not None:
        model.backbone.extra[0][7][4] = nn.Sequential(
            model.backbone.extra[0][7][4],
            attention_block(attention_transition_type, 1024)
        )  # after conv7 (post-ReLU)

    if attention_head_type is not None:
        cls_in_channels = [module.in_channels for module in model.head.classification_head.module_list]
        cls_attention = ApplyAttentionToList(attention_head_type, cls_in_channels)
        model.head.classification_head = nn.Sequential(
            cls_attention,
            model.head.classification_head
        )

        reg_in_channels = [module.in_channels for module in model.head.regression_head.module_list]
        reg_attention = ApplyAttentionToList(attention_head_type, reg_in_channels)
        model.head.regression_head = nn.Sequential(
            reg_attention,
            model.head.regression_head
        )

    return model


if __name__ == '__main__':
    print(ssd_ball_detector(True, None, 'se'))