from __future__ import division

from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from optimizers import Ranger
from data_parallel import DataParallel

from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", default='weights/yolov3.weights', \
                        type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_weights", default='weights/darknet53.conv.74', \
    #                     type=str, help="if specified starts from checkpoint model") # pretrained on ImageNet
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=0, help="number of GPUs to use")
    parser.add_argument("--input_len", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--dim", choices=[0,1], default=1, type=int, help=" ")
    parser.add_argument("--use_gauss", action='store_true', help=" ")
    opt = parser.parse_args()
    print(opt)

    # Prepare fundamentals
    logger = Logger("logs")
    device = torch.device("cuda" if opt.n_gpu else "cpu")
    print('n_GPUs: {}'.format(opt.n_gpu))
    print('Batch Size: {}'.format(opt.batch_size))
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
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

    '''
    ## Checks model with dummy input
    # Note: Pytorch Default is [B, C, "H", "W"]

    if opt.dim == 2: ## 2D
      B, C, W, H = opt.batch_size, 3, 416, 416
      inp = torch.rand((B, C, W, H), dtype=torch.float).to(device)

    elif opt.dim == 1: ## 1D
      B, C, W, H = opt.batch_size, 1, 416, 1
      inp = torch.rand((B, C, W), dtype=torch.float).to(device)

    out = model(inp)
    print(out)
    '''
    '''
    model_1d = Darknet(opt.model_def, dim=1)
    # Print learnable parameters
    names_1d = []
    params_1d = []
    params_1d_shape = []
    for name, param in model_1d.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            names_1d.append(name)
            params_1d.append(param.data)
            params_1d_shape.append(param.data.shape)

    model_2d = Darknet(opt.model_def, dim=2)
    # Print learnable parameters
    names_2d = []
    params_2d = []
    params_2d_shape = []
    for name, param in model_2d.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            names_2d.append(name)
            params_2d.append(param.data)
            params_2d_shape.append(param.data.shape)

    np.array(names_1d) == np.array(names_2d)

    for i in range(len(params_2d_shape)):
      # print(params_2d_shape[i], params_1d_shape[i])
      if len(params_2d_shape[i]) == 4:
        issame = (params_2d[i].mean(dim=-1).shape == params_1d_shape[i])
        if not issame:
          print(names_2d[i])

    '''

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )


    # Replicate Model if n_GPUs > 1
    if opt.n_gpu > 1:
      opt.batch_size *= opt.n_gpu
      device_ids = list(range(torch.cuda.device_count()))
      model = DataParallel(model, device_ids=device_ids, output_device=None)

    optimizer = Ranger(model.parameters(), lr=1e-3, eps=1e-8)

    ## Main Loop
    for epoch in range(opt.epochs):

        model.train()
        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            '''
            epoch = 0
            iterator = iter(enumerate(dataloader))
            batch_i, (_, imgs, targets) = next(iterator)
            '''

            batches_done = len(dataloader) * epoch + batch_i

            if opt.dim == 1:
              imgs = imgs.mean(axis=1).unsqueeze(1) # Convert RGB to Gray, [b, c, h, w]
              imgs = imgs.mean(axis=2) # Compress the Y dimension
              targets = targets[:, [0, 1, 2, 4]] # Exclude CentreY and Width
              targets[:,1,...] = 0 # Converts all classes to "person' to check the model as a pure object detector'

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False) # [n_bounding_ranges_in_the_batch, 4]

            '''
            # targets.shape[-1]
            # [Image idx in a batch, GT Class of bounding box, CentreX, (CentreY,) , Width (,Height)]

            ## Confirmation with plotting
            # Randomly choose a target
            i_target = np.random.choice(len(targets))
            target = targets[i_target].cpu()
            i_img, i_cls, X, Y, W, H = target
            img = imgs[i_img.to(torch.long)].cpu()
            cls_name = class_names[i_cls.to(torch.long)] # fixme

            ## Plot a picture
            npimg = img.numpy()
            npimg = np.transpose(npimg, (1,2,0))
            # plt.imshow(npimg, interpolation='nearest')
            # plt.show()

            ## Plot Bounding Box on the picture # fixme
            import cv2
            size = npimg.shape[0]
            X, Y, W, H = X*size, Y*size, W*size, H*size # Converts relative coordinates to absolute ones
            pt1 = (int(X - int(W/2)), int(Y - int(H/2)))
            pt2 = (int(X + int(W/2)), int(Y - int(H/2)))
            pt3 = (int(X + int(W/2)), int(Y + int(H/2)))

            dst = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
            dst = cv2.rectangle(dst, pt1, pt2, pt3)

            cv2.imshow(cls_name, dst)
            cv2.waitKey(5000)
            print(cls_name)

            cv2.destroyAllWindows()
            '''

            loss, outputs, metrics = model(imgs, targets=targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ---------------- #
            #   Log progress   #
            # ---------------- #
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers[0]))]]]
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            # Log metrics at each YOLO layer
            metrics_keys =  list(metrics[0].keys())
            formats = {m: "%.6f" for m in metrics_keys}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            for i, metric in enumerate(metrics_keys):
                row_metrics = []
                for i_yolo in range(len(model.yolo_layers)):
                    yolo_metrics = metrics[i_yolo]
                    row_metrics.append(formats[metric] % yolo_metrics.get(metric, 0))
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []

                for i_yolo in range(len(model.yolo_layers)):
                    yolo_metrics = metrics[i_yolo]
                    for name, metric in yolo_metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{i_yolo+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:

            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.001,
                nms_thres=0.5,
                img_size=opt.input_len,
                batch_size=8,
                dim=opt.dim,
            )

            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
