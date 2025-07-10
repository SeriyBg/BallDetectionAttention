import os
import pickle
import time

import torch
from torchvision.models.detection import ssd300_vgg16
from tqdm import tqdm

from data.ball_annotated_3k_yolov5_dataset_utils import make_dfl_dataloaders
from misc.config import Params


def ssd_ball_detector():
    # Create SSD model with 2 classes (background + ball)
    num_classes = 2  # background + ball
    model = ssd300_vgg16(num_classes=num_classes)
    return model

if __name__ == '__main__':
    print(ssd_ball_detector())