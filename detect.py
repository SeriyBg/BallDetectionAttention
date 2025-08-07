import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.augmentation import augmentations
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from data.detect_utils import draw_boxes
from models.train import model_factory
from parameter_rader import get_parameters

if __name__ == '__main__':
    params = get_parameters()

    model = model_factory(params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_ca_head_eca_20250804_1246_final.pth"
    state_dict = torch.load(model_path,
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

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

