import os

import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter, RandomAffine, RandomCrop, Normalize, \
    RandomHorizontalFlip

from data.augmentation import NORMALIZATION_MEAN, NORMALIZATION_STD
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from misc.config import Params


def make_dfl_dataloaders(params: Params):
    size = (1280, 720)
    train_transform = Compose([
        # ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # RandomAffine(degrees=5, scale=(0.8, 1.2)),
        # RandomHorizontalFlip(),
        Resize(size),
        ToTensor(),
        Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
    ])
    val_transform = Compose([
        Resize(size),
        ToTensor(),
        Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
    ])
    train_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=train_transform,
        mode="train",
        num_workers=params.num_workers,
    )
    val_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=val_transform,
        mode="valid",
        num_workers=params.num_workers,
    )

    dataloaders = {'train': DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                       collate_fn=lambda x: tuple(zip(*x))),
                   'val': DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))}
    return dataloaders


if __name__ == '__main__':
    # Dataset and transform
    transform = Compose([Resize((300, 300)), ToTensor()])
    dataset = BallAnnotated3kYOLOV5Dataset(
        root="/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5",
        transform=None,  # Don't transform for OpenCV visualization
        mode="train"
    )

    for idx in range(len(dataset)):
        image_pil, target = dataset[idx]
        image_path = os.path.join(dataset.images_dir, dataset.image_list[idx])
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print(f"[Warning] Could not read image: {image_path}")
            continue

        h, w, _ = image_bgr.shape

        if target["boxes"].nelement() == 0:
            print(f"[Info] No ball in frame: {image_path}")
        else:
            boxes = target["boxes"].numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(image_bgr, "ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Ball Annotation", image_bgr)

        if cv2.waitKey(0) & 0xFF == 27:  # Esc key to exit
            break

    cv2.destroyAllWindows()
