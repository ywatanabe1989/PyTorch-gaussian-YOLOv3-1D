from __future__ import division

from models import *
# from models import Darknet
from utils.utils import *
# from utils.utils import xywh2xyxy, non_max_suppression
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, dim=2):
    '''
    fixme
    1) evaluate() and dataloader should be independent. If dataset changed from COCO, you have to change this evaluate funciton, too.
    2) NMS 1D might have bugs because there were many indices to specify coordinates, confidence, and class probabilities
    '''
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        if opt.dim == 1: # fixme
          imgs = imgs.mean(axis=1).unsqueeze(1) # Convert RGB to Gray, [b, c, h, w]
          imgs = imgs.mean(axis=2) # Compress the Y dimension
          targets = targets[:, [0, 1, 2, 4]] # Exclude CentreY and Width
          targets[:,1,...] = 0 # Converts all classes to "person' to check the model as a pure object detector', fixme

        # Extract labels
        labels += targets[:, 1].tolist()

        with torch.no_grad():
            outputs = model(imgs)

            if dim == 2:
              outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            if dim == 1:
              outputs = non_max_suppression_1D(outputs, conf_thres=conf_thres, nms_thres=nms_thres) # fixme

        if dim == 2:
          sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        if dim == 1:
          sample_metrics += get_batch_statistics_1D(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", default='weights/yolov3.weights', \
                        type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_weights", default='weights/darknet53.conv.74', \
    #                     type=str, help="if specified starts from checkpoint model") # pretrained on ImageNet
    # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--dim", choices=[1, 2], default=1, type=int, help=" ")
    parser.add_argument("--use_gauss", action='store_true', help=" ")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def, dim=opt.dim, use_gauss=opt.use_gauss).to(device)
    model.apply(weights_init_normal)

    # Get learnable parameters
    learnable_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            learnable_param_names.append(name)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.dim == 2:
          if opt.pretrained_weights.endswith(".pth"):
              model.load_state_dict(torch.load(opt.pretrained_weights))
          else:
              model.load_darknet_weights(opt.pretrained_weights)

        if opt.dim == 1: # model is 1D
          _model = Darknet(opt.model_def, dim=2, use_gauss=opt.use_gauss).to(device) # _model is 2D
          if opt.pretrained_weights.endswith(".pth"):
              _model.load_state_dict(torch.load(opt.pretrained_weights))
          else:
              _model.load_darknet_weights(opt.pretrained_weights)

          # Convert 2D pretrained weights to 1D with mean
          params_2d = _model.state_dict()
          params_1d = model.state_dict()
          size_error = 0
          errored_names = []
          substitusion = 0
          for i, name in enumerate(learnable_param_names):
              param_2d = params_2d[name] # torch.Tensor
              param_1d = model.state_dict()[name] # torch.Tensor
              # Convert 2D weight to 1D
              if len(param_2d.shape) == len(param_1d.shape) + 1:
                  param_2d = param_2d.mean(dim=-1)
                  if param_2d.shape[1] == 3 and param_1d.shape[1] == 1: # the 1st layer for the channel difference
                      param_2d = param_2d.mean(dim=1).unsqueeze(1)
                  if param_2d.shape == param_1d.shape:
                      params_1d[name] = param_2d
                      substitusion += 1
                  else:
                      size_error += 1
                      errored_names.append(name) # YOLO Layers' weights are not inherited because of the num_classes difference.

          model.load_state_dict(params_1d)

          '''
          # Dry-run for checking
          for i, name in enumerate(learnable_param_names):
              param_2d = params_2d[name] # torch.Tensor
              param_1d = model.state_dict()[name] # torch.Tensor
              # Convert 2D weight to 1D
              if len(param_2d.shape) == len(param_1d.shape) + 1:
                  param_2d = param_2d.mean(dim=-1)
                  if param_2d.shape[1] == 3 and param_1d.shape[1] == 1: # the 1st layer for the channel difference
                      param_2d = param_2d.mean(dim=1).unsqueeze(1)
                  if param_2d.shape == param_1d.shape:
                      print(model.state_dict()[name] == param_2d)
                  else:
                      pass
          '''


    # # Initiate model
    # model = Darknet(opt.model_def, dim=opt.dim).to(device)
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        dim=opt.dim,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
