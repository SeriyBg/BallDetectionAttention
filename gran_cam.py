import cv2
import numpy as np
import requests
import torch
import torchvision
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from torchvision.transforms.v2 import Compose

from data.augmentation import BallCropTransform, ToTensor
from data.ball_annotated_3k_yolov5_dataset import BallAnnotated3kYOLOV5Dataset
from models.ssd_voc import ssd_ball_detector


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)[0]  # SSD returns List[Dict]
    pred_classes = [coco_names[i] for i in outputs['labels'].cpu().numpy()]
    pred_labels = outputs['labels'].cpu().numpy()
    pred_scores = outputs['scores'].detach().cpu().numpy()
    pred_bboxes = outputs['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


def ssd_reshape_transform(x):
    # SSD300 outputs feature maps from multiple layers for detection
    # So we use the last feature map (VGG-like feature extractor)
    # Grad-CAM will receive one tensor (not a dict like Faster R-CNN)
    return x


class SSDBoxScoreTarget:
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.tensor([0.0], device=model_outputs['scores'].device)
        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.tensor(box[None, :], device=model_outputs['boxes'].device)
            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output += score
        return output


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return image_with_bounding_boxes


# COCO labels
coco_names = ['__background__', 'ball']
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


if __name__ == '__main__':
    dataset = BallAnnotated3kYOLOV5Dataset(
        root="/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5",
        transform=Compose([
            BallCropTransform(300),
            ToTensor(),
        ]),
        mode="train",
    )

    # Get a sample from dataset
    image_tensor, target = dataset[0]  # Or any index

    image_float_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_float_np = np.clip(image_float_np, 0, 1)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = image_tensor.unsqueeze(0).to(device)

    model = ssd_ball_detector(attention=True, attention_backbone_type='eca', attention_head_type=None)
    model_path = "/Users/sergebishyr/PhD/models/ball_detection/100e/ssd_backbone_eca_20250803_0544_final.pth"
    state_dict = torch.load(
        model_path,
        map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    boxes, classes, labels, indices = predict(input_tensor, model, device, detection_threshold=0.4)

    target_layers = [model.backbone.features[-4], model.backbone.features[-2]]  # Last conv layer of backbone
    # target_layers = [model.head.regression_head.module_list[2]]  # First conv layer of regression head
    targets = [SSDBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

    # cam = EigenCAM(model,
    #                target_layers,
    #                reshape_transform=ssd_reshape_transform)
    # cam = GradCAMPlusPlus(model, target_layers, reshape_transform=ssd_reshape_transform)
    cam = AblationCAM(model, target_layers, reshape_transform=ssd_reshape_transform)

    grayscale_cam = cam(input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    Image.fromarray(image_with_bounding_boxes).show()
    # image_with_bounding_boxes = renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam)
    # Image.fromarray(image_with_bounding_boxes).show()
