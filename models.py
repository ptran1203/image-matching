import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet
import gc
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
from rexnetv1 import ReXNetV1
from resnest.torch import resnest101
from util import l2_norm
from tqdm import tqdm
from transformers import AutoModel

class EffnetV2(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained=True, bert=None, loss_type='aarc'):
        super(EffnetV2, self).__init__()
        enet_type = enet_type.replace('-', '_')

        feat_dim = 512
        planes = self._get_global_dim(enet_type)
        self.enet = geffnet.create_model(enet_type,
            pretrained=pretrained, as_sequential=True)[:-4]

        if bert is not None:
            self.bert = AutoModel.from_pretrained('/content/bert-base-uncased')
            planes += self.bert.config.hidden_size
        else:
            self.bert = None

        # self.feat = nn.Linear(self.enet.classifier.in_features, feat_dim)
        # self.swish = Swish_module()
        if loss_type == 'aarc':
            self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)
        elif loss_type == 'arc':
            self.arc = ArcModule(feat_dim, out_dim)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift

        self.gem = GeM()


    def forward(self, x, input_ids=None, attention_mask=None):
        x = self.enet(x)
        global_feat = self.gem(x)
        global_feat = global_feat.view(global_feat.size()[0], -1)
        if self.bert is not None:
            text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
            x = torch.cat([x, text], 1)

        feat = self.to_feat(global_feat)
        feat = self.bn(feat)
        feat = l2_norm(feat, axis=-1)

        logits_m = self.arc(feat)
        return feat, logits_m

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
    def __init__(self, backbones, folds, stages, loss_types, weight_dir, out_dim=11014, reduction='mean'):
        super(EnsembleModels, self).__init__()

        self.backbones = backbones
        self.folds = folds
        self.stages = stages
        self.loss_types = loss_types
        self.weight_dir = weight_dir
        self.out_dim = out_dim
        self.reduction = reduction  # mean or concat
        self.models = self.load_models()

    def load_effnets(self, backbone, fold, stage, loss_type):
        weight_path = os.path.join(self.weight_dir, f'{backbone}_fold{fold}_stage{stage}_{loss_type}.pth')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'{weight_path} does not exist')

        model = EffnetV2(backbone, out_dim=self.out_dim, pretrained=False, loss_type=loss_type)
        model = model.cuda()
        checkpoint = torch.load(weight_path, map_location='cuda:0')
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        return model

    def load_models(self):
        max_len = max([len(p) for p in [self.backbones, self.folds, self.stages, self.loss_types] if isinstance(p, list)] + [1])

        if not isinstance(self.backbones, list):
            self.backbones = [self.backbones] * max_len

        if not isinstance(self.folds, list):
            self.folds = [self.folds] * max_len

        if not isinstance(self.stages, list):
            self.stages = [self.stages] * max_len

        if not isinstance(self.loss_types, list):
            self.loss_types = [self.loss_types] * max_len

        models = []
        for backbone, fold, stage, loss_type in zip(self.backbones, self.folds, self.stages, self.loss_types):
            print(f'Loading model {backbone} - fold {fold} - stage {stage} - loss {loss_type}')
            model = self.load_effnets(backbone, fold, stage, loss_type)
            model.eval()
            models.append(model)

        return models

    def forward(self, x):
        results = []
        for model in self.models:
            feat, _ = model(x)
            results.append(feat)

        if len(results) == 1:
            return results[0]
        
        if self.reduction == 'concat':
            return torch.cat(results, dim=-1)
        elif self.reduction == 'mean':
            return torch.mean(torch.stack(results), dim=0)

        return results


def inference(model, test_loader, tqdm=tqdm):
    embeds = []
    is_ensemble = isinstance(model, EnsembleModels)
    if not is_ensemble:
        model.eval()
    with torch.no_grad():
        for i, img in enumerate(tqdm(test_loader)):
            img = img.cuda()
            feat = model(img)
            if not is_ensemble:
                feat = feat[1]

            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
        
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    return image_embeddings


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
        cosine_all = F.linear(features, F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, scale=30, margin=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = torch.tensor(math.cos(math.pi - margin))
        self.mm = torch.tensor(math.sin(math.pi - margin) * margin)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs