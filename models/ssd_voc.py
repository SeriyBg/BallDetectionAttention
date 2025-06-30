import os
import time

import cv2
import torch
from networkx.algorithms.shortest_paths.unweighted import predecessor
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from data.detect_utils import predict, draw_boxes
from misc.config import Params

from PIL import Image

MODEL_FOLDER = 'saved_models'


def ssd_ball_detector(params: Params):
    # Prepare dataset
    transform = Compose([Resize((300, 300)), ToTensor()])
    train_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=transform,
        mode="train",
        num_workers=params.num_workers,
    )
    val_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=transform,
        mode="valid",
        num_workers=params.num_workers,
    )
    dataloaders = {'train': DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x))),
                   'val': DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))}

    # Create SSD model with 2 classes (background + ball)
    model = ssd300_vgg16(weights="DEFAULT")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)

    # Training loop
    num_epochs = params.epochs
    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        for phase, dataloader in dataloaders.items():
            # if phase == 'train':
            #     model.train()
            # else:
            #     model.eval()
            model.train()
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

            print(f"{phase} [SSD] - Loss: {total_loss:.4f}")

    model_name = 'ssd_' + time.strftime("%Y%m%d_%H%M")
    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

if __name__ == '__main__':
    ssd_ball_detector(Params("../config.txt"))