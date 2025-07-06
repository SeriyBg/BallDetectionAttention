import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from tqdm import tqdm

from data.augmentation import NORMALIZATION_MEAN, NORMALIZATION_STD
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from data.ball_annotated_3k_yolov5_dataset_utils import make_dfl_dataloaders
from misc.config import Params

MODEL_FOLDER = 'saved_models'


def ssd_ball_detector():
    # Create SSD model with 2 classes (background + ball)
    num_classes = 2  # background + ball
    model = ssd300_vgg16(num_classes=2)
    model.transform.min_size = (1920, 1080)
    model.transform.max_size = (1920, 1080)
    return model


def train_ssd(params: Params):
    dataloaders = make_dfl_dataloaders(params)
    # Create SSD model with 2 classes (background + ball)
    num_classes = 2  # background + ball
    model = ssd_ball_detector()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # if torch.mps.device_count() > 0:
    #     device = "mps"
    model.to(device)
    # Training loop
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)  # momentum=0.9, weight_decay=5e-4)
    scheduler_milestones = [int(params.epochs * 0.25), int(params.epochs * 0.50), int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)

    num_epochs = params.epochs
    training_stats = {'train': [], 'val': []}
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for phase, dataloader in dataloaders.items():
            # if phase == 'train':
            #     model.train()
            # else:
            #     model.eval()
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                    # model.eval()
                    # image = Image.open("/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5_2/valid/images/3c993bd2_0_58_play_png.rf.05e540bcf9a4bc3e434abe43dbaf25b3.jpg")
                    # boxes, classes, labels = predict(image, model, device, 0.5)
                    # image = draw_boxes(boxes, classes, labels, image)
                    # cv2.imshow('Image', image)
                    # cv2.waitKey(0)

                    optimizer.zero_grad()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()

            training_stats[phase].append({'total_loss': total_loss, 'loc_loss': loss_dict['bbox_regression'].item(), 'cls_loss': loss_dict['classification'].item()})
            print(f"{phase} [SSD] - Loss: {total_loss:.4f}; Loc Loss: {loss_dict['bbox_regression']:.4f}; Cls Loss: {loss_dict['classification']:.4f}")
        scheduler.step()

    model_name = 'ssd_' + time.strftime("%Y%m%d_%H%M")
    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

if __name__ == '__main__':
    print(ssd_ball_detector())