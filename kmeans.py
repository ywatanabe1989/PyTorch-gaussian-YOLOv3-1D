from __future__ import division

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

from models import *
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"] # trainvalno5kpart
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=False, multiscale=opt.multiscale_training) # augment=True
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False, # True, fixme
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    '''
    ## Pytorch Default
    [B, C, H, W]

    ## 2D (Original)
    n_ch = 3
    w, h = 416, 416
    bs = opt.batch_size
    Xb = torch.rand((bs, n_ch, w, h), dtype=torch.float).to(device) # (0, 1]
    output = model(Xb)

    ## 1D
    n_ch = 1
    w, h = 416, 1
    bs = opt.batch_size
    Xb = torch.rand((bs, n_ch, w), dtype=torch.float).to(device) # (0, 1]
    output = model(Xb)
    '''

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    kmeans = MiniBatchKMeans(n_clusters=9, batch_size=opt.batch_size)



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

            # imgs = Variable(imgs.to(device)) # [4, 3, 384, 384]
            # targets = Variable(targets.to(device), requires_grad=False) # [n_bboxes_in_the_batch, 6]
            kmeans = kmeans.partial_fit(targets[:, 4:])
            centers = kmeans.cluster_centers_
            print(np.sort(centers*imgs.shape[-1], axis=0).astype(np.int))




'''
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 0], [4, 4],
              [4, 5], [0, 1], [2, 2],
              [3, 2], [5, 5], [1, -1]])

# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters=2,
                         random_state=0,
                         batch_size=6)

kmeans = kmeans.partial_fit(X[0:6,:])
kmeans = kmeans.partial_fit(X[6:12,:])
centers = kmeans.cluster_centers_
# array([[1, 1],
#        [3, 4]])
## Plot
plt.scatter(X[:,0], X[:,1], label='X')
plt.scatter(centers[:,0], centers[:,1], label='centers')
plt.legend()
plt.show()




kmeans.predict([[0, 0], [4, 4]])
# array([0, 1], dtype=int32)
# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=2,
                         random_state=0,
                         batch_size=6,
                         max_iter=10).fit(X)
centers = kmeans.cluster_centers_
# array([[3.95918367, 2.40816327],
#        [1.12195122, 1.3902439 ]])
kmeans.predict([[0, 0], [4, 4]])
# array([1, 0], dtype=int32)

## Plot
plt.scatter(X[:,0], X[:,1], label='X')
plt.scatter(centers[:,0], centers[:,1], label='centers')
plt.legend()
plt.show()
'''
