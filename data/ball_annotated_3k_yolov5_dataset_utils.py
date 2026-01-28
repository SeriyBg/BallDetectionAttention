import cv2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms.v2 import Compose

from data.augmentation import BallCropTransform, ToTensor, augmentations, BallColorJitter, RandomAffine
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from misc.config import Params


def make_dfl_dataloaders(params: Params):
    # size = (1280, 720)
    dfl_train = []
    dfl_val = []
    for _, dfl_path in enumerate(params.dfl_paths):
        dfl_train.append(BallAnnotated3kYOLOV5Dataset(
            root=dfl_path,
            transform=augmentations(params),
            mode="train",
            num_workers=params.num_workers,
            ball_labels=params.ball_labels,
        ))
        dfl_val.append(BallAnnotated3kYOLOV5Dataset(
            root=dfl_path,
            transform=augmentations(params),
            mode="valid",
            num_workers=params.num_workers,
            ball_labels=params.ball_labels,
        ))
    train_dataset = ConcatDataset(dfl_train) if len(dfl_train) > 1 else dfl_train[0]
    val_dataset = ConcatDataset(dfl_val) if len(dfl_val) > 1 else dfl_val[0]

    dataloaders = {'train': DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                       collate_fn=lambda x: tuple(zip(*x))),
                   'val': DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))}
    return dataloaders


if __name__ == '__main__':
    # Base dataset (no transform — we'll operate on raw image for visualization)
    dataset = BallAnnotated3kYOLOV5Dataset(
        # root="/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5",
        root="/Users/sergebishyr/PhD/datasets/Detect Players.v7i.yolov5pytorch",
        transform=Compose([
            RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5)),
            BallCropTransform(300),
            BallColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ToTensor(),
        ]),
        mode="train",
        ball_labels=["Ball"]
    )

    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]  # image_tensor: (C, H, W)
        image_np = image_tensor.permute(1, 2, 0).numpy()  # (H, W, C), float [0,1]
        image_bgr = (image_np * 255).astype("uint8")
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

        if target["boxes"].nelement() == 0:
            print(f"[Info] No ball in cropped frame at idx {idx}")
        else:
            boxes = target["boxes"].numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(image_bgr, "ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Cropped Ball Annotation (300x300)", image_bgr)

        if cv2.waitKey(0) & 0xFF == 27:  # Esc key to exit
            break

    cv2.destroyAllWindows()
