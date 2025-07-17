import argparse

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.augmentation import BallCropTransform, ToTensorAndNormalize, Resize
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
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

    model_path = "/Users/sergebishyr/PhD/models/ball_detection/ssd_attention_crop_300_7aa39cdbadd65be59321ec520834dcf77e680497/ssd_20250713_1652_final.pth"
    model_path = "/Users/sergebishyr/PhD/models/ball_detection/fasterrcnn_eef54c9cec1a9bfcdee987e0d4d5a9aad34678c9fcbd05/ssd_20250715_1256_final.pth"
    model_path = "/Users/sergebishyr/PhD/models/ball_detection/fasterrcnn_attention_eef54c9cec1a9bfcdee987e0d4d5a9aad34678c9fcbd05/ssd_20250715_1622_final.pth"
    state_dict = torch.load(model_path,
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    ds = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=Compose([
            # Resize((720, 1280)),
            BallCropTransform(720),
            ToTensorAndNormalize()
        ]),
        mode="test",
        num_workers=params.num_workers,
    )
    dl = DataLoader(ds, batch_size=params.batch_size, shuffle=True,
               collate_fn=lambda x: tuple(zip(*x)))
    for images, _ in dl:

        outputs = model(images)
        for image, prediction in zip(images, outputs):
            boxes, labels, scores = prediction["boxes"], prediction["labels"], prediction["scores"]
            image = draw_boxes(image, boxes, labels, scores, 0.4)
            cv2.imshow('Image', image)
            if cv2.waitKey(0) & 0xFF == 27:  # Esc key to exit
                break
        else:
            continue
        break

    cv2.destroyAllWindows()

