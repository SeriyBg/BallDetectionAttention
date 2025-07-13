import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from data.augmentation import NORMALIZATION_MEAN, NORMALIZATION_STD, ToTensorAndNormalize, BallCropTransform, ToTensor
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from data.ball_crop_dataset import BallCropWrapperDataset
from misc.config import Params


def make_dfl_dataloaders(params: Params):
    # size = (1280, 720)
    train_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=Compose([BallCropTransform(), ToTensorAndNormalize()]),
        mode="train",
        num_workers=params.num_workers,
    )
    val_dataset = BallAnnotated3kYOLOV5Dataset(
        root=params.dfl_path,
        transform=Compose([BallCropTransform(), ToTensorAndNormalize()]),
        mode="valid",
        num_workers=params.num_workers,
    )

    dataloaders = {'train': DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                       collate_fn=lambda x: tuple(zip(*x))),
                   'val': DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))}
    return dataloaders


if __name__ == '__main__':
    # Base dataset (no transform — we'll operate on raw image for visualization)
    dataset = BallAnnotated3kYOLOV5Dataset(
        root="/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5",
        transform=Compose([
            BallCropTransform(),
            ToTensor(),
            # ToTensorAndNormalize(),
        ]),
        mode="train"
    )

    # Wrap it with 300x300 ball-aware crop
    # dataset = BallCropWrapperDataset(base_dataset,
    #                                  transform=None,
    #                                  crop_size=512)

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
