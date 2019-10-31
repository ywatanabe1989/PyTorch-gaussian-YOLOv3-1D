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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch") # 512
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/yolov3_1d.cfg", help="path to model definition file")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=4, help="number of GPUs to use")
    parser.add_argument("--use_fp16", action='store_true', help=" ")
    parser.add_argument("--input_len", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)
    use_opti = 'ranger'
    init_lr = 1e-3

    # Switches dimensions
    if '1d' in opt.model_def: # 1D
      dim = 1; from models_1d import *
    else: # 2D
      dim = 2; from models import *

    logger = Logger("logs")

    device = torch.device("cuda" if opt.n_gpu else "cpu")

    opt.batch_size *= opt.n_gpu
    print('n_GPUs: {}'.format(opt.n_gpu))

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    learnable_params = model.parameters()

    '''
    ## Model Confirmation with dummy mini-batch input

    # Note: Pytorch Default is [B, C, "H", "W"]

    if dim == 2: ## 2D
      B, C, W, H = opt.batch_size, 3, 416, 416
      inp = torch.rand((B, C, W, H), dtype=torch.float).to(device)

    elif dim == 1: ## 1D
      B, C, W, H = opt.batch_size, 1, 416, 1
      inp = torch.rand((B, C, W), dtype=torch.float).to(device)

    out = model(inp)
    print(out)
    '''

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

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
      device_ids = list(range(torch.cuda.device_count()))
      model = DataParallel(model, device_ids=device_ids, output_device=None) # nn.parallel.DataParallel(model)
      # replicas = nn.parallel.replicate(model, device_ids)
      # for i_gpu, replica in enumerate(replicas):
      #   # replica.to(torch.device("cuda:{}".format(i)))
      #   replica.cuda(i_gpu)

      # model = replicas[0]
      # del model
      # learnable_params = []
      # for replica in replicas:
      #     learnable_params += list(replica.parameters())
      # learnable_params = list(replicas[0].parameters()) \
      #                  + list(replicas[1].parameters()) \
      #                  + list(replicas[2].parameters()) \
      #                  + list(replicas[3].parameters())

    # opt.use_fp16 = False
    # if opt.use_fp16:
    #   from apex import amp
    #   adam_eps = 1e-4
    #   if use_opti == 'ranger':
    #     optimizer = Ranger(model.parameters(), lr=init_lr, eps=adam_eps)
    #   if use_opti == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, nesterov=False)
    #   amp.register_float_function(torch, 'sigmoid')
    #   model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=1)
    # else:
    adam_eps = 1e-8
    # if use_opti == 'ranger':
    # optimizer = Ranger(model.parameters(), lr=init_lr, eps=adam_eps)
    # if use_opti == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, nesterov=False)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "z",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    if dim == 1:
      metrics.pop(metrics.index('y'))
      metrics.pop(metrics.index('z'))

    for epoch in range(opt.epochs):

        # if opt.n_gpu > 1:
        #   for replica in replicas:
        #       replica.train()
        # else:
        #    model.train()
        model.train()

        start_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            '''
            epoch = 0
            iterator = iter(enumerate(dataloader))
            batch_i, (_, imgs, targets) = next(iterator)
            '''

            batches_done = len(dataloader) * epoch + batch_i

            if dim == 1:
              imgs = imgs.mean(axis=1).unsqueeze(1) # Convert RGB to Gray, [b, c, h, w]
              imgs = imgs.mean(axis=2) # Compress the Y dimension
              targets = targets[:, [0, 1, 2, 4]] # Exclude CentreY and Width
              targets[:,1,...] = 0 # Converts all classes to "person' to check the model as a pure object detector'

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False) # [n_bounding_ranges_in_the_batch, 4]

            if opt.n_gpu:
                # n_coordinates = targets.shape[-1] - 2
                bs_per_gpu = int( len(imgs) / opt.n_gpu)
                targets_list = []
                for i_gpu in range(opt.n_gpu):
                  targets_for_a_gpu_mask = (bs_per_gpu*i_gpu <= targets[:,0]) & (targets[:,0] < bs_per_gpu*(i_gpu+1))
                  targets_for_a_gpu = targets[targets_for_a_gpu_mask]
                  targets_for_a_gpu[:,0] -= bs_per_gpu*i_gpu
                  targets_list.append({'targets':targets_for_a_gpu.cuda(i_gpu)})

                # targets_reshaped_list = [targets_list[i].reshape(-1) for i in range(opt.n_gpu)]
                # padded_targets = torch.nn.utils.rnn.pad_sequence(targets_reshaped_list, batch_first=True, padding_value=np.nan)
                # targets = {'targets':padded_targets}

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

            if opt.n_gpu > 1:
                scattered_imgs = nn.parallel.scatter(imgs, device_ids)
                kwargs_tup = tuple(targets_list)
                # loss, outputs = nn.parallel.parallel_apply(replicas, imgs, kwargs_tup=kwargs_tup)
                replicas_outputs = model(scattered_imgs, kwargs_tup=kwargs_tup)
                # replicas_outputs = nn.parallel.parallel_apply(replicas, scattered_imgs, kwargs_tup=kwargs_tup)

                replicas_losses = [replicas_outputs[i_gpu][0] for i_gpu in range(opt.n_gpu)]
                outputs = np.vstack([replicas_outputs[i_gpu][1] for i_gpu in range(opt.n_gpu)])
                for loss in replicas_losses:
                  loss.backward()
            else:
              loss, outputs = model(imgs, targets=targets)
              loss.backward()

            # if opt.use_fp16:
            #   with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            # else:
            #   loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            yolo_layers = model.module.yolo_layers if opt.n_gpu > 1 else model.yolo_layers

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                # row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []

                # for j, yolo in enumerate(model.yolo_layers):
                for j, yolo in enumerate(yolo_layers):
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

            # model.seen += imgs.size(0)
            if opt.n_gpu > 1:
              model.module.seen += imgs.size(0)
            else:
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
                img_size=opt.input_len,
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
