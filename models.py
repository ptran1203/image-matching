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
import gc
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
        # self.swish = Swish_module()
        self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift

        self.gem = GeM()


    def forward(self, x):
        x = self.enet(x)

        # global_feat = F.adaptive_avg_pool2d(x, 1)
        global_feat = self.gem(x)
        # global_feat = F.avg_pool2d(x, x.size()[2:])
        global_feat = global_feat.view(global_feat.size()[0], -1)
        # global_feat = F.dropout(global_feat, p=0.2)

        feat = self.to_feat(global_feat)
        feat = self.bn(feat)
        logits_m = self.arc(feat)
        feat = l2_norm(feat, axis=-1)

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


class EnsembleModels(nn.Module):
    def __init__(self, backbones, folds, stages, weight_dir, out_dim=11014, reduction='mean'):
        super(EnsembleModels, self).__init__()

        self.backbones = backbones
        self.folds = folds
        self.stages = stages
        self.weight_dir = weight_dir
        self.out_dim = out_dim
        self.reduction = reduction  # mean or concat
        self.models = self.load_models()

    def load_effnets(self, backbone, fold, stage):
        weight_path = os.path.join(self.weight_dir, f'{backbone}_fold{fold}_stage{stage}.pth')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'{weight_path} does not exist')

        model = EffnetV2(backbone, out_dim=self.out_dim, pretrained=False)
        model = model.cuda()
        checkpoint = torch.load(weight_path, map_location='cuda:0')
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        return model

    def load_models(self):
        max_len = max([len(p) for p in [self.backbones, self.folds, self.stages] if isinstance(p, list)] + [1])

        if not isinstance(self.backbones, list):
            self.backbones = [self.backbones] * max_len

        if not isinstance(self.folds, list):
            self.folds = [self.folds] * max_len

        if not isinstance(self.stages, list):
            self.stages = [self.stages] * max_len

        models = []
        for backbone, fold, stage in zip(self.backbones, self.folds, self.stages):
            model = self.load_effnets(backbone, fold, stage)

        return models

    def forward(self, x):
        results = []
        for model in self.models:
            pred = model(x)
            results.append(pred)

        if self.reduction == 'concat':
            return torch.cat(results, dim=0)
        elif self.reduction == 'mean':
            return torch.mean(results)

        return results


def inference(model, test_loader):
    embeds = []

    with torch.no_grad():
        for img in tqdm_notebook(test_loader): 
            img = img.cuda()
            feat, _ = model(img)

            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
        
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    return image_embeddings
