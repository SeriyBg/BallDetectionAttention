import os
import pickle
import time

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from data.ball_annotated_3k_yolov5_dataset_utils import make_dfl_dataloaders
from misc.config import Params
from models.fasterrcnn import fasterrccn_mobilnet, fasterrccn
from models.ssd_voc import ssd_ball_detector

MODEL_FOLDER = 'saved_models'


def train(params: Params):
    dataloaders = make_dfl_dataloaders(params)
    model = model_factory(params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if hasattr(torch.mps, "device_count"):
    #     if torch.mps.device_count() > 0:
    #         device = "mps"
    print(f"Using device: {device}")
    model.to(device)
    # Training loop
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=5e-4)
    scheduler_milestones = [int(params.epochs * 0.25), int(params.epochs * 0.50), int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)

    num_epochs = params.epochs
    training_stats = {'train': [], 'val': []}
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for phase, dataloader in dataloaders.items():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                    optimizer.zero_grad()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()

            if params.model == 'ssd':
                training_stats[phase].append({'total_loss': total_loss, 'loc_loss': loss_dict['bbox_regression'].item(), 'cls_loss': loss_dict['classification'].item()})
                print(f"{phase} [SSD] - Loss: {total_loss:.4f}; Loc Loss: {loss_dict['bbox_regression']:.4f}; Cls Loss: {loss_dict['classification']:.4f}")
            elif params.model == 'fasterrcnn' or params.model == 'fasterrcnn_mobilenet':
                training_stats[phase].append({'total_loss': total_loss} | loss_dict)
                print(f"{phase} [SSD] - Loss: {total_loss:.4f}; Loc Loss: {loss_dict['loss_box_reg']:.4f}; Cls Loss: {loss_dict['loss_classifier']:.4f}; RPN cls loss: {loss_dict['loss_objectness']:.4f}, RPN loc loss: {loss_dict['loss_rpn_box_reg']:.4f}")
        scheduler.step()

    model_name = 'ssd_' + time.strftime("%Y%m%d_%H%M")
    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)


def model_factory(params: Params):
    model = None
    if params.model == 'ssd':
        model = ssd_ball_detector(params.attention)
    elif params.model == 'fasterrcnn':
        model = fasterrccn(params)
    elif params.model == 'fasterrcnn_mobilenet':
        model = fasterrccn_mobilnet(params)
    assert model is not None, 'Unknown model type: {}'.format(params.model)
    print(model)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return model
