import random

import numpy as np
import torch
from torch.utils.data import Dataset


class BallCropWrapperDataset(Dataset):
    def __init__(self, base_dataset, transform, crop_size=300):
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]

        # Get image as tensor (C, H, W) and original dimensions
        if isinstance(image, torch.Tensor):
            img_tensor = image
        else:
            raise TypeError("Base dataset must return image as a tensor")

        _, H, W = img_tensor.shape

        if len(target["boxes"]) == 0:
            # No ball in image — fallback to center crop or skip
            crop_top = max(0, (H - self.crop_size) // 2)
            crop_left = max(0, (W - self.crop_size) // 2)
        else:
            # Use the first ball box
            box = target["boxes"][0]
            x_min, y_min, x_max, y_max = box.tolist()
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            # Calculate crop bounds so the ball center is inside
            crop_left_min = max(0, int(cx) - self.crop_size + 1)
            crop_left_max = min(int(cx), W - self.crop_size)
            crop_top_min = max(0, int(cy) - self.crop_size + 1)
            crop_top_max = min(int(cy), H - self.crop_size)

            if crop_left_max < crop_left_min or crop_top_max < crop_top_min:
                # Fallback if ball too close to edge
                crop_left = min(max(0, int(cx) - self.crop_size // 2), W - self.crop_size)
                crop_top = min(max(0, int(cy) - self.crop_size // 2), H - self.crop_size)
            else:
                crop_left = random.randint(crop_left_min, crop_left_max)
                crop_top = random.randint(crop_top_min, crop_top_max)

        # Crop the image
        cropped_img = img_tensor[:, crop_top:crop_top+self.crop_size, crop_left:crop_left+self.crop_size]
        if self.transform:
            cropped_img = self.transform(cropped_img)

        # Adjust boxes
        new_boxes = []
        new_labels = []
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            x1_new = x1 - crop_left
            y1_new = y1 - crop_top
            x2_new = x2 - crop_left
            y2_new = y2 - crop_top

            # Check if box is still inside crop
            if x2_new <= 0 or y2_new <= 0 or x1_new >= self.crop_size or y1_new >= self.crop_size:
                continue  # box fully outside

            # Clip box to crop boundaries
            x1_new = np.clip(x1_new, 0, self.crop_size)
            y1_new = np.clip(y1_new, 0, self.crop_size)
            x2_new = np.clip(x2_new, 0, self.crop_size)
            y2_new = np.clip(y2_new, 0, self.crop_size)

            new_boxes.append([x1_new, y1_new, x2_new, y2_new])
            new_labels.append(label)

        if new_boxes:
            new_target = {
                "boxes": torch.tensor(new_boxes, dtype=torch.float32),
                "labels": torch.tensor(new_labels, dtype=torch.int64)
            }
        else:
            new_target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            }

        return cropped_img, new_target
