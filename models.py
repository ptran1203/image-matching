import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import geffnet
from rexnetv1 import ReXNetV1
from resnest.torch import resnest101
from util import l2_norm

class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class Effnet(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained=True):
        super(Effnet, self).__init__()
        feat_dim = 512
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=pretrained)
        self.feat = nn.Linear(self.enet.classifier.in_features, feat_dim)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(feat_dim, out_dim)
        self.enet.classifier = nn.Identity()

        self.feature_bn = nn.BatchNorm1d(feat_dim)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.swish(self.feat(x))
        x = self.feature_bn(x)
        logits_m = self.metric_classify(x)

        return l2_norm(x, axis=-1), logits_m


class RexNet20(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained=True):
        super(RexNet20, self).__init__()
        self.enet = ReXNetV1(width_mult=2.0)
        if pretrained:
            pretrain_wts = "./rexnetv1_2.0x.pth"
            sd = torch.load(pretrain_wts)
            self.enet.load_state_dict(sd, strict=True)
        
        self.feat = nn.Linear(self.enet.output[1].in_channels, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.output = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.swish(self.feat(x))
        logits_m = self.metric_classify(x)

        return l2_norm(x, axis=-1), logits_m


class ResNest101(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained=True):
        super(ResNest101, self).__init__()
        self.enet = resnest101(pretrained=pretrained)
        self.feat = nn.Linear(self.enet.fc.in_features, 512)
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.fc = nn.Identity()
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.swish(self.feat(x))
        logits_m = self.metric_classify(x)

        return l2_norm(x, axis=-1), logits_m
