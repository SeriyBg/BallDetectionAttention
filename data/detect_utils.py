import torch
import torchvision.transforms.v2 as transforms
import cv2
import numpy as np

from data.augmentation import NORMALIZATION_STD, NORMALIZATION_MEAN

coco_names = [
    '__background__', 'ball'
]

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    image = image.unsqueeze(0)
    # get the predictions on the image
    outputs = model(image)

    # get all the predicited class names
    pred_classes = [
        coco_names[i] if 0 <= i < len(coco_names) else f"__background__"
        for i in outputs[0]['labels'].cpu().numpy()
    ]

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    return boxes, pred_classes, outputs[0]['labels']


def draw_boxes(image, boxes, labels, scores, detection_threshold):
    """
    Draws the bounding box around a detected object.
    """
    image = denormalize(image, mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
    image = np.asarray(image.permute(1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # classes = [coco_names[int(i)] for i in labels]
    labels = labels.detach().numpy()
    boxes = boxes.detach().numpy()
    classes = [
        coco_names[int(i)] if 0 <= i < len(coco_names) else f"__background__"
        for i in labels
    ]
    for i, box in enumerate(boxes):
        if scores[i] < detection_threshold:
            continue
        color = COLORS[int(labels[i])]
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


def denormalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): shape (C, H, W), normalized
        mean (list): mean used in normalization
        std (list): std used in normalization
    Returns:
        Tensor: denormalized tensor, still (C, H, W)
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean
