import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision

from data.augmentation import augmentations
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from misc.config import Params
from models.train import model_factory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    args = parser.parse_args()

    models = [
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_20250802_0816_final.pth", "section" : "DEFAULT"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_ca_20250803_0209_final.pth", "section" : "BACK_CA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_eca_20250803_0544_final.pth", "section" : "BACK_ECA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_se_20250802_2231_final.pth", "section" : "BACK_SE"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_head_ca_20250802_1522_final.pth", "section" : "HEAD_CA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_head_eca_20250802_1859_final.pth", "section" : "HEAD_ECA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_head_se_20250802_1147_final.pth", "section" : "HEAD_SE"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_ca_head_ca_20250804_0718_final.pth", "section" : "BACK_CA_HEAD_CA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_ca_head_eca_20250804_1246_final.pth", "section" : "BACK_CA_HEAD_ECA"},
        {"path": "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_ca_head_se_20250804_1808_final.pth", "section" : "BACK_CA_HEAD_SE"},
    ]

    print('Config path: {}'.format(args.config))
    results = {}
    for m in models:
        params = Params(args.config, m["section"])
        model = model_factory(params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        dfl_test = []
        for _, dfl_path in enumerate(params.dfl_paths):
            dfl_test.append(BallAnnotated3kYOLOV5Dataset(
                root=dfl_path,
                transform=augmentations(params),
                mode="test",
                num_workers=params.num_workers,
                ball_labels=params.ball_labels,
            ))
        ds = ConcatDataset(dfl_test) if len(dfl_test) > 1 else dfl_test[0]
        dl = DataLoader(ds, batch_size=params.batch_size, shuffle=True,
                        collate_fn=lambda x: tuple(zip(*x)))

        map_metric = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])  # VOC-style mAP@0.5

        # Run inference and collect predictions + ground truth
        all_preds = []
        all_targets = []

        state_dict = torch.load(
            m["path"],
            map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        with torch.no_grad():
            iter = 1
            while iter <= 10:
                for images, targets in dl:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    outputs = model(images)

                    for pred, gt in zip(outputs, targets):
                        all_preds.append({
                            "boxes": pred["boxes"].detach().cpu(),
                            "scores": pred["scores"].detach().cpu(),
                            "labels": pred["labels"].detach().cpu()
                        })
                        all_targets.append({
                            "boxes": gt["boxes"].detach().cpu(),
                            "labels": gt["labels"].detach().cpu()
                        })
                iter += 1

        # Compute mAP
        map_metric.update(all_preds, all_targets)
        result = map_metric.compute()
        results[m['section']] = result["map_50"]

    for key, value in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{key}: {value:.4f}")


