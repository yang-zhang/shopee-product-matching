import os
import cv2
import glob
import math
import pickle
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import albumentations
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import geffnet


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


sigmoid = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class enet_arcface_FINAL(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(enet_arcface_FINAL, self).__init__()
        self.enet = geffnet.create_model(enet_type.replace("-", "_"), pretrained=None)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)


from rexnetv1 import ReXNetV1
from resnest.torch import resnest101


class rex20_arcface(nn.Module):
    def __init__(self, enet_type, out_dim, load_pretrained=False):
        super(rex20_arcface, self).__init__()
        self.enet = ReXNetV1(width_mult=2.0)
        if load_pretrained:
            pretrain_wts = "/workspace/rexnetv1_2.0x.pth"
            sd = torch.load(pretrain_wts)
            self.enet.load_state_dict(sd, strict=True)

        self.feat = nn.Linear(self.enet.output[1].in_channels, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.output = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)


class nest101_arcface(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(nest101_arcface, self).__init__()
        self.enet = resnest101(pretrained=False)
        self.feat = nn.Linear(self.enet.fc.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.fc = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)


# def load_model(model, model_file):
#     state_dict = torch.load(model_file)
#     if "model_state_dict" in state_dict.keys():
#         state_dict = state_dict["model_state_dict"]
#     state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
# #     del state_dict['metric_classify.weight']
#     model.load_state_dict(state_dict, strict=True)
#     print(f"loaded {model_file}")
#     model.eval()
#     return model


def load_model(model, plmodel):
    state_dict = plmodel.state_dict()
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    state_dict = {
        k[6:] if k.startswith("model.") else k: state_dict[k] for k in state_dict.keys()
    }
    #     del state_dict['metric_classify.weight']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
