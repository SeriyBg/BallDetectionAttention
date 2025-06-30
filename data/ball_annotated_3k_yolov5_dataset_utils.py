import os

import cv2
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset

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
