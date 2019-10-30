from __future__ import division

from models_1d import *
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
from apex import amp
from apex.parallel import DistributedDataParallel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="size of each image batch") # 512
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_1d.cfg", help="path to model definition file")
    # parser.add_argument("--model_def", type=str, default="config/yolov3-tiny_1d.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--input_len", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # n_gpus = int(torch.cuda.device_count())
    # if n_gpus > 1: # fixme
    #   local_rank = 0
    #   torch.cuda.set_device(local_rank)
    #   torch.distributed.init_process_group(backend='nccl', init_method='env://')

    opt.batch_size *= opt.n_gpu #
    print('n_GPUs: {}'.format(opt.n_gpu))

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    model_config = parse_model_config(opt.model_def)
    train_path = data_config["train"] # trainvalno5kpart
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)

    # for name, param in model.named_parameters():
    #   if param.device.type != 'cuda':
    #       print('param {}, not on GPU'.format(name))

    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training) # augment=True, fixme

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )

    '''
    ## Pytorch Default
    [B, C, H, W]

    ## 1D
    n_ch = 1
    w, h = 416, 1
    bs = opt.batch_size
    Xb = torch.rand((bs, n_ch, w), dtype=torch.float).to(device) # (0, 1]
    output = model(Xb)
    '''

    # optimizer = torch.optim.Adam(model.parameters()) # fixme

    use_fp16 = True # fixme
    use_opti = 'ranger' # fixme
    init_lr = 1e-3 # fixme
    num_losses = 1

    if use_fp16:
      adam_eps = 1e-4
      if use_opti == 'ranger':
        optimizer = Ranger(model.parameters(), lr=init_lr, eps=adam_eps)
      if use_opti == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, nesterov=False)
      amp.register_float_function(torch, 'sigmoid')
      model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=num_losses)
    else:
      adam_eps = 1e-8
      if use_opti == 'ranger':
        optimizer = Ranger(model.parameters(), lr=init_lr, eps=adam_eps)
      if use_opti == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, nesterov=False)

    if opt.n_gpu > 1:
      # model = torch.nn.DataParallel(model).to(device)
      # model = DistributedDataParallel(model).to(device)
      device_ids = list(range(torch.cuda.device_count()))
      replicas = nn.parallel.replicate(model, device_ids)
      for i,replica in enumerate(replicas):
        replica.to(torch.device("cuda:{}".format(i)))
      del model

    metrics = [
        "grid_size",
        "loss",
        "x",
        "w",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):

        if opt.n_gpu > 1:
          for replica in replicas:
              replica.train()
        else:
           model.train()
        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            '''
            epoch = 0
            iterator = iter(enumerate(dataloader))
            batch_i, (_, imgs, targets) = next(iterator)
            '''
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.mean(axis=1).unsqueeze(1) # Convert RGB to Gray, [b, c, h, w]
            imgs = imgs.mean(axis=2) # Compress the Y dimension
            imgs = Variable(imgs.to(device))
            targets = targets[:, [0, 1, 2, 4]] # Exclude CentreY and Width
            targets[:,1,...] = 0 # Converts all classes to "person' to check the model as a pure object detector'
            targets = Variable(targets.to(device), requires_grad=False) # [n_bounding_ranges_in_the_batch, 4]

            if opt.n_gpu:
                n_coordinates = targets.shape[-1] - 2
                bs = len(imgs)
                bs_sep = int(bs / opt.n_gpu)
                targets_list = []
                for i_sep in range(opt.n_gpu):
                  targets_sep_mask = (bs_sep*i_sep <= targets[:,0]) & (targets[:,0] < bs_sep*(i_sep+1))
                  targets_sep = targets[targets_sep_mask]
                  targets_sep[:,0] -= bs_sep*i_sep
                  targets_list.append({'targets':targets_sep.cuda(i_sep)})

                # targets_reshaped_list = [targets_list[i].reshape(-1) for i in range(opt.n_gpu)]
                # padded_targets = torch.nn.utils.rnn.pad_sequence(targets_reshaped_list, batch_first=True, padding_value=np.nan)
                # targets = {'targets':padded_targets}

            '''
            targets
            1 Image number in a batch
            2 Class of bounding box
            3 CentreX
            4 CnetreY
            5 Width
            6 Height

            # Choose a target
            i_target = np.random.choice(len(targets))
            target = targets[i_target].cpu()
            i_img, i_cls, X, Y, W, H = target
            img = imgs[i_img.to(torch.long)].cpu()
            cls = class_names[i_cls.to(torch.long)] # fixme
            print(cls)

            ## Plot Original Picture
            npimg = img.numpy()
            npimg = np.transpose(npimg, (1,2,0))
            plt.imshow(npimg, interpolation='nearest')
            plt.show()

            ## Plot Bounding Box on the original pic
            #     import cv2
            #     CentreX, CentreY, W, H = target[2:]
            #     pt1 = (int(CentreX - int(W/2)), int(CentreY - int(H/2)))
            #     pt2 = (int(CentreX + int(W/2)), int(CentreY - int(H/2)))
            #     pt3 = (int(CentreX + int(W/2)), int(CentreY + int(H/2)))
            #     npimg = cv2.rectangle(npimg, pt1, pt2, pt3)
            '''

            if opt.n_gpu > 1:
                imgs = nn.parallel.scatter(imgs, device_ids)
                loss, outputs = nn.parallel.parallel_apply(replicas, imgs, kwargs_tup=tuple(targets_list))
            else:
              loss, outputs = model(imgs, targets=targets)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
              scaled_loss.backward()

            # loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
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
                conf_thres=0.5,
                nms_thres=0.5,
                input_len=opt.input_len,
                batch_size=8,
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
