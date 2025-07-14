import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import Compose
from tqdm import tqdm

from data.augmentation import BallCropTransform, ToTensorAndNormalize
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
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

    ds = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=Compose([BallCropTransform(), ToTensorAndNormalize()]),
        mode="test",
        num_workers=params.num_workers,
    )
    dl = DataLoader(ds, batch_size=params.batch_size, shuffle=True,
                    collate_fn=lambda x: tuple(zip(*x)))

    map_metric = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])  # VOC-style mAP@0.5

    # Run inference and collect predictions + ground truth
    all_preds = []
    all_targets = []

    model_path = "/Users/sergebishyr/PhD/models/ball_detection/ssd_crop_300_7aa39cdbadd65be59321ec520834dcf77e680497/ssd_20250713_1405_final.pth"
    model_path = "/Users/sergebishyr/PhD/models/ball_detection/ssd_attention_crop_300_7aa39cdbadd65be59321ec520834dcf77e680497/ssd_20250713_1652_final.pth"
    state_dict = torch.load(
        model_path,
        map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, targets in tqdm(dl, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for pred, gt in zip(outputs, targets):
                all_preds.append({
                    "boxes": pred["boxes"].detach().to(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu()
                })
                all_targets.append({
                    "boxes": gt["boxes"].detach().cpu(),
                    "labels": gt["labels"].detach().cpu()
                })

    # Compute mAP
    map_metric.update(all_preds, all_targets)
    results = map_metric.compute()
    print(f"mAP@0.5: {results['map_50']:.4f}")
    print(f"mAP (COCO-style): {results['map']:.4f}")
    print(f"Precision: {results['map_per_class']}")
    print(f"Recall: {results['mar_100']:.4f}")
