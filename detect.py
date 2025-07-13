import argparse

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor, Compose

from data.augmentation import NORMALIZATION_MEAN, NORMALIZATION_STD
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from data.ball_crop_dataset import BallCropWrapperDataset
from data.detect_utils import draw_boxes
from misc.config import Params
from models.train import model_factory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    params = Params(args.config)

    model = model_factory(params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "/Users/sergebishyr/PhD/Courses/BallDetectionAttention/saved_models/ssd_20250709_1357_final.pth"
    model_path = "/Users/sergebishyr/PhD/models/ball_detection/ssd_no_resize_dfl.pth"
    state_dict = torch.load(model_path,
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    ds = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=Compose([ToTensor(), Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)]),
        mode="test",
        num_workers=params.num_workers,
    )
    ds = BallCropWrapperDataset(ds, transform=None)
    dl = DataLoader(ds, batch_size=params.batch_size, shuffle=True,
               collate_fn=lambda x: tuple(zip(*x)))
    for images, _ in dl:

        outputs = model(images)
        for image, prediction in zip(images, outputs):
            boxes, labels, scores = prediction["boxes"], prediction["labels"], prediction["scores"]
            image = draw_boxes(image, boxes, labels, scores, 0.5)
            cv2.imshow('Image', image)
            if cv2.waitKey(0) & 0xFF == 27:  # Esc key to exit
                break
        else:
            continue
        break

    cv2.destroyAllWindows()

