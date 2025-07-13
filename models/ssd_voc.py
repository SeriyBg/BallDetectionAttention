import os
import pickle
import time

import torch
from torch import nn
from torchvision.models.detection import ssd300_vgg16
from tqdm import tqdm

from data.ball_annotated_3k_yolov5_dataset_utils import make_dfl_dataloaders
from misc.config import Params
from models.attention import SEBlock


def ssd_ball_detector(attention: bool = False):
    # Create SSD model with 2 classes (background + ball)
    num_classes = 2  # background + ball
    if attention:
        return ssd_ball_detector_attention(num_classes)
    return ssd300_vgg16(num_classes=num_classes)

def ssd_ball_detector_attention(num_classes):
    model = ssd300_vgg16(num_classes=num_classes)
    vgg_features = model.backbone.features

    vgg_features[15] = nn.Sequential(vgg_features[15], SEBlock(256))  # after conv3_3
    vgg_features[22] = nn.Sequential(vgg_features[22], SEBlock(512))  # after conv4_3
    model.backbone.features = vgg_features
    return model


if __name__ == '__main__':
    print(ssd_ball_detector(True))