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
from torchvision.models.resnet import ResNet
from rexnetv1 import ReXNetV1
from resnest.torch import resnest101, resnest50
from util import l2_norm
from tqdm import tqdm
from transformers import AutoModel
from util import search_weight, scale_img
# import timm

root_dir = '/content' if os.path.exists('/content') else '/kaggle/input'


class EffnetV2(nn.Module):

    def __init__(self, enet_type, out_dim, pretrained=True, bert=False, loss_type='aarc'):
        super(EffnetV2, self).__init__()
        enet_type = enet_type.replace('-', '_')

        feat_dim = 512
        planes = self._get_global_dim(enet_type)
        self.enet = geffnet.create_model(enet_type,
            pretrained=pretrained, as_sequential=True)[:-4]

        if bert:
            self.bert = AutoModel.from_pretrained(f'{root_dir}/bert-base-uncased')
            planes += self.bert.config.hidden_size
        else:
            self.bert = None

        # self.feat = nn.Linear(self.enet.classifier.in_features, feat_dim)
        # self.swish = Swish_module()
        if loss_type == 'aarc':
            self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)
        elif loss_type == 'arc':
            self.arc = ArcModule(feat_dim, out_dim)
        elif loss_type == 'cos':
            self.arc = CosModule(feat_dim, out_dim)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift

        self.gem = GeM()


    def forward(self, x, input_ids, attention_mask, labels=None):
        x = self.enet(x)
        global_feat = self.gem(x)
        global_feat = global_feat.view(global_feat.size()[0], -1)

        if self.bert is not None:
            text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
            global_feat = torch.cat([global_feat, text], 1)

        feat = self.to_feat(global_feat)
        feat = self.bn(feat)
        feat = l2_norm(feat, axis=-1)

        if labels is not None:
            logits_m = self.arc(feat, labels)
        else:
            logits_m = None
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


class Resnest50(nn.Module):

    def __init__(self, out_dim, pretrained=True, bert=False, loss_type='aarc'):
        super(Resnest50, self).__init__()

        feat_dim = 512
        self.backbone = torch.nn.Sequential(*list(resnest50(pretrained=True).children())[:-2])
        planes = 2048
        if bert:
            self.bert = AutoModel.from_pretrained(f'{root_dir}/bert-base-uncased')
            planes += self.bert.config.hidden_size
        else:
            self.bert = None

        self.gem = GeM()

        if loss_type == 'aarc':
            self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)
        elif loss_type == 'arc':
            self.arc = ArcModule(feat_dim, out_dim)
        elif loss_type == 'cos':
            self.arc = CosModule(feat_dim, out_dim)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        # self.bn.bias.requires_grad_(False)  # no shift

    def forward(self, x, input_ids=None, attention_mask=None, labels=None):
        x = self.backbone(x)
        global_feat = self.gem(x)
        global_feat = global_feat.view(global_feat.size()[0], -1)

        if self.bert is not None:
            text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
            global_feat = torch.cat([global_feat, text], 1)

        feat = self.to_feat(global_feat)
        feat = self.bn(feat)
        feat = l2_norm(feat, axis=-1)

        if labels is not None:
            logits_m = self.arc(feat, labels)
        else:
            logits_m = None
        return feat, logits_m


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
    def __init__(self, backbones, folds, stages, loss_types, weight_dir, reduction='mean', tta=False, out_dim=False):
        super(EnsembleModels, self).__init__()

        self.backbones = backbones
        self.folds = folds
        self.stages = stages
        self.loss_types = loss_types
        self.weight_dir = weight_dir
        self.reduction = reduction  # mean or concat
        self.tta = tta  # E.g ['hflip', '']
        self.out_dim = out_dim
        self.models = self.load_models()

    def load_effnets(self, backbone, fold, stage, loss_type):
        weight_path = search_weight(self.weight_dir, backbone, fold, stage, loss_type)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f'{weight_path} does not exist')
        
        if not self.out_dim:
            self.out_dim = self.get_outdim(weight_path)
            # overring outdim
            
            weight_path = weight_path.split("outdim") + f"outdim{elf.out_dim}.pth"

        print(weight_path)
        if backbone == 'auto':
            backbone = self.get_backbone(weight_path)

        print(f'Loading model {backbone} - fold {fold} - stage {stage} - loss {loss_type}, dim {self.out_dim}')
        if backbone == 'resnest50':
            model = Resnest50(out_dim=self.out_dim, pretrained=False, loss_type=loss_type)
        else:
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
            model = self.load_effnets(backbone, fold, stage, loss_type)
            model.eval()
            models.append(model)

        return models

    def forward(self, img, input_ids, att_mask):
        results = []
        for model in self.models:
            if self.tta:
                tta_preds = []
                for trans_img in self.get_TTA(img):
                    feat, _ = model(trans_img, input_ids, att_mask)
                    tta_preds.append(feat)

                mean_pred = torch.mean(torch.stack(tta_preds), dim=0)
                results.append(l2_norm(mean_pred))
            else:
                feat, _ = model(img, input_ids, att_mask)
                results.append(feat)

        if len(results) == 1:
            return results[0]

        if self.reduction in {'cat', 'concat'}:
            return l2_norm(torch.cat(results, dim=-1))
        elif self.reduction == 'mean':
            return l2_norm(torch.mean(torch.stack(results), dim=0))

        return torch.stack(results)

    @staticmethod
    def get_TTA(img):
        return [
            img,
            img.flip(3),  # Hflip
            scale_img(img, 0.8),
        ]

    @staticmethod
    def get_outdim(path):
        return int(path.split(".")[0].split("outdim")[1])

    @staticmethod
    def get_backbone(path):
        return path.split('/')[-1].split('_fold')[0]


def inference(model, test_loader, tqdm=tqdm):
    embeds = []
    is_ensemble = isinstance(model, EnsembleModels)
    if not is_ensemble:
        model.eval()
    with torch.no_grad():
        for i, (img, input_ids, att_mask) in enumerate(tqdm(test_loader)):
            img, input_ids, att_mask = img.cuda(), input_ids.cuda(), att_mask.cuda()
            feat = model(img, input_ids, att_mask)
            if not is_ensemble:
                feat = feat[1]

            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    gc.collect()
    return np.concatenate(embeds)


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
        
    def forward(self, features, labels):
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

class CosModule(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, scale=30.0, margin=0.40):
        super(CosModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
