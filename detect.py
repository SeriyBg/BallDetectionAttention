import argparse

import cv2
import torch
from PIL import Image

from data.detect_utils import predict, draw_boxes
from models.ssd_voc import ssd_ball_detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))

    model = ssd_ball_detector()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dict = torch.load("/Users/sergebishyr/PhD/models/ssd_ball_detection/ssd_no_resize_dfl.pth",
                            map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    image = Image.open("/Users/sergebishyr/PhD/datasets/ball_annotated_3k_yolov5/test/images/1606b0e6_1_352_play_png.rf.22db27e0331fb4d1c396c3fbb57da832.jpg")
    boxes, classes, labels = predict(image, model, device, 0.5)
    image = draw_boxes(boxes, classes, labels, image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)