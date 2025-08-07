import os

import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image


class BallAnnotated3kYOLOV5Dataset(Dataset):
    def __init__(self, root, transform, mode, num_workers=1, ball_labels=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_dir = os.path.join(root, mode, 'images')
        self.labels_dir = os.path.join(root, mode, 'labels')
        self.image_list = [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        self.num_workers =num_workers
        if ball_labels is None:
            self.ball_labels = []
        else:
            with open(root + '/data.yaml', 'r') as file:
                yaml_data = yaml.safe_load(file)
            # Access the specific key
            all_labels = yaml_data['names']
            self.ball_labels = [i for i, x in enumerate(all_labels) if x in ball_labels]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_filename = self.image_list[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, image_filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    l, xc, yc, w, h = map(float, parts)

                    # Convert YOLO (normalized) to pixel (xmin, ymin, xmax, ymax)
                    xc *= width
                    yc *= height
                    w *= width
                    h *= height

                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2

                    if self.ball_labels is None:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # 1 = ball
                    elif l in self.ball_labels:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # 1 = ball

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
        else:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }

        if self.transform:
            image, target = self.transform((image, target))

        return image, target

