#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import sys
from torchsummary import summary
# sys.path.append('/home/po/TM5/Pointnet_Pointnet2_pytorch')
sys.path.append('/home/po/TM5/s4g-release/inference/grasp_proposal/network_models')
sys.path.append('/home/po/TM5/graspnetAPI/graspnetAPI')
sys.path.append('/home/po/TM5')
from nn_utils.mlp import SharedMLP
from pointnet2_utils.modules import PointNetSAModule, PointnetFPModule, PointNetSAAvgModule
from nn_utils.functional import smooth_cross_entropy
sys.path


# In[ ]:


sys.path.append('/home/po/TM5/graspnet-baseline')
import scipy.io as scio
from dataset.graspnet_dataset1 import GraspNetDataset, collate_fn, load_grasp_labels
from PIL import Image


# In[ ]:


class PointNet2(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

    def __init__(self,
                 score_classes,
                 num_centroids=(10240, 1024, 128, 0),
                 radius=(0.2, 0.3, 0.4, -1.0),
                 num_neighbours=(64, 64, 64, -1),
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 512, 1024)),
                 fp_channels=((256, 256), (256, 128), (128, 128), (64, 64, 64)),
                 num_fp_neighbours=(0, 3, 3, 3),
                 seg_channels=(128,),
                 num_removal_directions=5,
                 dropout_prob=0.5):
        super(PointNet2, self).__init__()

        # Sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius) == num_sa_layers
        assert len(num_neighbours) == num_sa_layers
        assert len(sa_channels) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(num_fp_neighbours) == num_fp_layers

        # Set Abstraction Layers
        feature_channels = 0
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [0]
        inter_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            fp_module = self._FP_MODULE(in_channels=feature_channels + inter_channels[-2 - ind],
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
#         self.mlp_seg = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
#         self.seg_logit = nn.Conv1d(seg_channels[-1], score_classes, 1, bias=True)

        self.mlp_grasp_eval = SharedMLP(feature_channels + 28, seg_channels, ndim=2, dropout_prob=dropout_prob)
        self.grasp_eval_logit = nn.Conv2d(seg_channels[-1], 1, 1, bias=True)
    
        self.mlp_R = SharedMLP(feature_channels, seg_channels, ndim=1)
        self.R_logit = nn.Conv1d(seg_channels[-1], 4, 1, bias=True)

        self.mlp_t = SharedMLP(feature_channels, seg_channels, ndim=1)
        self.t_logit = nn.Conv1d(seg_channels[-1], 3, 1, bias=True)

#         self.mlp_movable = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
#         self.movable_logit = nn.Sequential(
#             nn.Conv1d(seg_channels[-1], num_removal_directions, 1, bias=True),
#             nn.Sigmoid())

        self.init_weights()
    def forward(self, data_batch):
        points = data_batch['point_clouds']

        xyz = points
        feature = None

        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [feature]

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

        # MLP
#         x = self.mlp_seg(sparse_feature)
#         logits = self.seg_logit(x)
        
              
        
        
        R = self.mlp_R(sparse_feature)
        R = self.R_logit(R)
        R = F.normalize(R, dim=1)
        # R = toRotMatrix(R)
        # R = euler2RotMatrix(R)

        t = self.mlp_t(sparse_feature)
        t = self.t_logit(t)
        
        local_search_frame = torch.cat([R, t], dim=1).unsqueeze(-1)
        local_search_frame = local_search_frame.repeat(1, 4, 1, 1)
        sparse_feature = sparse_feature.unsqueeze(-1)
        valid_feature = torch.cat([sparse_feature, local_search_frame], dim=1)
        local_search_logit = self.grasp_eval_logit(self.mlp_grasp_eval(valid_feature))
        # t = points + t
        
#         mov = self.mlp_movable(sparse_feature)
#         mov = self.movable_logit(mov)  # (B, 5, N)
        
            
        preds = {
#                 "score": logits,
                "score_pred": local_search_logit,
                 "q_pred": R,
                 "xyz_pred": t,
#                  "movable_logits": mov,
                 }

        return preds

    def init_weights(self):
        # nn_utils.init.zeros_(self.t_logit.weight)
        # nn_utils.init.zeros_(self.t_logit.bias)
        pass


# In[ ]:


class PointNet2Loss(nn.Module):
    def __init__(self, label_smoothing=0, neg_weight=0.1):
        super(PointNet2Loss, self).__init__()
        self.label_smoothing = label_smoothing
        self.neg_weight = neg_weight

    def forward(self, preds, labels):
        

        q_label = labels["q_label"].permute(0,2,1)#(1,4,20)
        q_pred = preds["q_pred"]
        q_loss = ((q_pred - q_label) ** 2).mean(1, True)
        

        # weight loss according to gt_score
        score_label = labels["score_label"].permute(0,2,1).unsqueeze(-1)
        score_pred = preds["score_pred"]
        score_loss = ((score_pred - score_label) ** 2).mean(1, True)
        

        xyz_label = labels["xyz_label"].permute(0,2,1)
        xyz_pred = preds["xyz_pred"]
        xyz_loss = ((xyz_pred - xyz_label) ** 2).mean(1, True)
        
    

        loss_dict = {
                    "score_loss": score_loss,
                     "q_loss": q_loss,
                     "xyz_loss": xyz_loss,
                     }

        return loss_dict


# In[ ]:


class PointNet2Metric(nn.Module):
    def forward(self, preds, labels):
        scene_score_logits = preds["scene_score_logits"]  # (B, C, N2)
        score_classes = scene_score_logits.shape[1]

        scene_score_labels = labels["scene_score_labels"]  # (B, N)

        selected_preds = scene_score_logits.argmax(1).view(-1)
        scene_score_labels = scene_score_labels.view(-1)

        cls_acc = selected_preds.eq(scene_score_labels).float()

        movable_logits = preds["movable_logits"]
        movable_labels = labels["scene_movable_labels"]
        movable_preds = (movable_logits > 0.5).view(-1).int()
        movable_labels = movable_labels.view(-1).int()
        mov_acc = movable_preds.eq(movable_labels).float()

        gt_frame_R = labels["best_frame_R"]
        batch_size, _, num_frame_points = gt_frame_R.shape
        pred_frame_R = preds["frame_R"][:, :, :num_frame_points]
        gt_frame_R = gt_frame_R.transpose(1, 2).contiguous().view(batch_size * num_frame_points, 3, 3)
        gt_frame_R_inv = gt_frame_R.clone()
        gt_frame_R_inv[:, :, 1:] = -gt_frame_R_inv[:, :, 1:]
        pred_frame_R = pred_frame_R.transpose(1, 2).contiguous().view(batch_size * num_frame_points, 3, 3)
        M = torch.bmm(gt_frame_R, pred_frame_R.transpose(1, 2))
        angle = torch.acos(torch.clamp((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0))
        M_inv = torch.bmm(gt_frame_R_inv, pred_frame_R.transpose(1, 2))
        angle_inv = torch.acos(torch.clamp((M_inv[:, 0, 0] + M_inv[:, 1, 1] + M_inv[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0))

        angle_min = torch.stack([angle, angle_inv], dim=1).min(1)[0]
        gt_scene_score = labels["scene_score"][:, :num_frame_points].contiguous().view(-1)
        angle_min = (gt_scene_score * angle_min).mean()

        gt_frame_t = labels["best_frame_t"].view(-1)
        pred_frame_t = preds["frame_t"][:, :, :num_frame_points]
        pred_frame_t = torch.argmax(pred_frame_t, dim=1).view(-1)
        t_acc = pred_frame_t.eq(gt_frame_t).float()

        # t_err = torch.mean(torch.sqrt(((gt_frame_t - pred_frame_t) ** 2).sum(1)))

        return {"cls_acc": cls_acc,
                "mov_acc": mov_acc,
                "R_err": angle_min,
                "t_acc": t_acc,
                }


# In[ ]:


# def build_pointnet2_cls(cfg):
def build_model(cfg):
    net = PointNet2(
        score_classes=cfg.DATA.SCORE_CLASSES,
        num_centroids=cfg.MODEL.PN2.NUM_CENTROIDS,
        radius=cfg.MODEL.PN2.RADIUS,
        num_neighbours=cfg.MODEL.PN2.NUM_NEIGHBOURS,
        sa_channels=cfg.MODEL.PN2.SA_CHANNELS,
        fp_channels=cfg.MODEL.PN2.FP_CHANNELS,
        num_fp_neighbours=cfg.MODEL.PN2.NUM_FP_NEIGHBOURS,
        seg_channels=cfg.MODEL.PN2.SEG_CHANNELS,
        num_removal_directions=cfg.DATA.NUM_REMOVAL_DIRECTIONS,
        dropout_prob=cfg.MODEL.PN2.DROPOUT_PROB,
    )

    loss_func = PointNet2Loss(
        label_smoothing=cfg.MODEL.PN2.LABEL_SMOOTHING,
        neg_weight=cfg.MODEL.PN2.NEG_WEIGHT,
    )
    metric = PointNet2Metric()

    return net, loss_func, metric



root = '/home/po/TM5/graspnetAPI'
valid_obj_idxs, grasp_labels = load_grasp_labels(root)
train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, split='train', remove_outlier=True, remove_invisible=True, num_points=20000)
print(len(train_dataset))




# train.py
import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# In[ ]:


TRAIN_DATALOADER = DataLoader(train_dataset, batch_size=1, shuffle=True,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)


# proposal_test.py
import sys,os
sys.path.append('/home/po/TM5/s4g-release/inference')
import numpy as np
import open3d
import time
import torch
import torch.nn as nn
from grasp_proposal.cloud_processor.cloud_processor import CloudPreProcessor
from grasp_proposal.configs.yacs_config import load_cfg_from_file
# from grasp_proposal.network_models.models.build_model import build_model
from grasp_proposal.utils.checkpoint import CheckPointer
from grasp_proposal.utils.file_logger_cls import loggin_to_file
from grasp_proposal.utils.grasp_visualizer import GraspVisualizer
from grasp_proposal.utils.logger import setup_logger, MetricLogger


# proposal_test.py
# load_static batch data
# 
cfg_path = "/home/po/TM5/s4g-release/inference/grasp_proposal/configs/curvature_model.yaml"
cfg = load_cfg_from_file(cfg_path)
cfg.defrost()
# cfg.TEST.WEIGHT = cfg.TEST.WEIGHT.replace("${PROJECT_HOME}", os.path.join(os.getcwd(), "../"))
# cfg.TEST.WEIGHT = '/home/po/TM5/s4g-release/inference/trained_models/curvature_model.pth'
cfg.TEST.WEIGHT = cfg.TEST.WEIGHT.replace("${PROJECT_HOME}", os.path.join('/home/po/TM5/s4g-release/inference'))
cfg.freeze()
assert cfg.TEST.BATCH_SIZE == 1

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

logger = setup_logger("S4G", output_dir, "unit_test")
logger.info("Using {} of GPUs".format(torch.cuda.device_count()))
logger.info("Load config file from {}".format(cfg_path))
logger.debug("Running with config \n {}".format(cfg))

model, loss, _ = build_model(cfg)


batch_data_label = next(iter(TRAIN_DATALOADER))
print("fuck")
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
print("fuck1")

model.eval()
print("fuck2")
data_batch = {
    "point_clouds": batch_data_label['point_clouds'].permute(0,2,1).float().cuda(),
    "xyz_label": batch_data_label['xyz_label'].float().cuda(),
    "q_label": batch_data_label['q_label'].float().cuda(),
    "score_label": batch_data_label['score_label'].float().cuda(),
    }
print("fuck3")
predictions = model(data_batch)
print("fuck4")
print(predictions)
# for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
#     data_batch = {
#     "point_clouds": batch_data_label['point_clouds'].float().cuda(),
#     "xyz_label": batch_data_label['xyz_label'].float().cuda(),
#     "q_label": batch_data_label['q_label'].float().cuda(),
#     "score_label": batch_data_label['score_label'].float().cuda(),
#     }
    # predictions = model(data_batch)
    # print(predictions)

    # loss = loss(predictions, data_batch).cuda()
    # sum(loss.values()).backward()   
    # optimizer.step()
    # optimizer.zero_grad()
    # print(loss['xyz_loss'],loss['q_loss'],loss['score_loss'])

