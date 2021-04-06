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
        self.feature_bn.bias.requires_grad_(False)  # no shift

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.swish(self.feat(x))
        x = self.feature_bn(x)
        logits_m = self.metric_classify(x)

        return l2_norm(x, axis=-1), logits_m


class EffnetV2(nn.Module):
    @staticmethod
    def _get_global_dim(enet_type):
        if 'b0' in enet_type:
            return 1208
        elif 'b1' in enet_type:
            return 1280
        elif 'b2' in enet_type:
            return 1408
        elif 'b3' in enet_type:
            return 1536
        elif 'b4' in enet_type:
            return 1792
        elif 'b5' in enet_type:
            return 2048
        elif 'b6' in enet_type:
            return 2304
        elif 'b7' in enet_type:
            return 2560

    def __init__(self, enet_type, out_dim, pretrained=True):
        super(EffnetV2, self).__init__()
        enet_type = enet_type.replace('-', '_')

        feat_dim = 512
        planes = self._get_global_dim(enet_type)
        self.enet = geffnet.create_model(enet_type,
            pretrained=pretrained, as_sequential=True)[:-4]
        # self.feat = nn.Linear(self.enet.classifier.in_features, feat_dim)
        self.swish = Swish_module()
        self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bottleneck_g = nn.BatchNorm1d(planes)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift


    def forward(self, x):
        x = self.enet(x)

        global_feat = F.adaptive_avg_pool2d(x, 1)
        # global_feat = F.avg_pool2d(x, x.size()[2:])
        global_feat = global_feat.view(global_feat.size()[0], -1)
        global_feat = F.dropout(global_feat, p=0.2)
        # global_feat = self.bottleneck_g(global_feat)
        # global_feat = l2_norm(global_feat, axis=-1)

        feat = self.to_feat(global_feat)

        feat = l2_norm(feat, axis=-1)

        logits_m = self.arc(feat)

        return feat, logits_m


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


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'