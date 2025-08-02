import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from data.augmentation import augmentations
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from models.train import model_factory
from parameter_rader import get_parameters

if __name__ == '__main__':
    params = get_parameters()

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
        ))
    ds = ConcatDataset(dfl_test) if len(dfl_test) > 1 else dfl_test[0]
    dl = DataLoader(ds, batch_size=params.batch_size, shuffle=True,
                    collate_fn=lambda x: tuple(zip(*x)))

    map_metric = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])  # VOC-style mAP@0.5

    # Run inference and collect predictions + ground truth
    all_preds = []
    all_targets = []

    model_path = "/Users/sergebishyr/PhD/models/ball_detection/ssd_attention_head_crop_300_94998f749d7ebe7a70028d28d146f201c68527ec/ssd_20250730_1248_final.pth"
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
                    "boxes": pred["boxes"].detach().cpu(),
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
