import math
import os
import numpy as np
import re
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
from util import search_weight, scale_img, freeze_bn
from losses import decode_config, encode_config
import timm

root_dir = '/content' if os.path.exists('/content') else '/kaggle/input'


class Model(nn.Module):

    def __init__(self, model_name, out_dim, pretrained=True,
        loss_config={}, args={},
    ):
        super(Model, self).__init__()

        fc_dim = 512
        n_classes = out_dim

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name.startswith('tf_efficientnet'):
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'nfnet_f3':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = not args.global_feat

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale = loss_config.scale,
            margin = loss_config.margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label=None):
        feature = self.extract_feat(image)

        if label is not None:
            logits = self.final(feature, label)
        else:
            logits = None

        return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x

class Resnest50(nn.Module):

    def __init__(self, out_dim, pretrained=True, loss_config={}, args={}):
        super(Resnest50, self).__init__()

        feat_dim = 512
        self.backbone = torch.nn.Sequential(*list(resnest50(pretrained=pretrained).children())[:-2])
        self.args = args
        planes = 2048
        if self.args.bert:
            self.bert = AutoModel.from_pretrained(f'{root_dir}/bert-base-uncased')
            planes += self.bert.config.hidden_size
        else:
            self.bert = None

        if self.args.global_feat:
            feat_dim = planes

        # self.pooling = GeM()
        self.pooling = nn.AdaptiveAvgPool2d(1)

        if self.args.freezebn:
            print(f'Freeze {freeze_bn(self.backbone)} layers')

        if loss_config.loss_type == 'aarc':
            self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)
        elif loss_config.loss_type == 'arc':
            self.arc = ArcModule(feat_dim, out_dim, scale=loss_config.scale, margin=loss_config.margin)
        elif loss_config.loss_type == 'cos':
            self.arc = CosModule(feat_dim, out_dim, scale=loss_config.scale, margin=loss_config.margin)

        self.to_feat = nn.Linear(planes, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)


    def forward(self, x, labels=None):
        x = self.backbone(x)
        feat = self.pooling(x)
        feat = feat.view(feat.size()[0], -1)

        if not self.args.global_feat:
            feat = self.to_feat(feat)
            feat = self.bn(feat)

        if labels is not None:
            logits_m = self.arc(feat, labels)
        else:
            logits_m = None
        return feat, logits_m

    def _init_params(self):
        nn.init.xavier_normal_(self.to_feat.weight)
        nn.init.constant_(self.to_feat.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


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
    def __init__(self, weight_list, weight_dir, reduction='mean', tta=False, args={}):
        super(EnsembleModels, self).__init__()

        self.weight_list = weight_list
        self.weight_dir = weight_dir
        self.args = args
        self.reduction = reduction  # mean or concat
        self.tta = tta  # E.g ['hflip', '']
        self.models = self.load_models()

    def load_effnets(self, weight_path):
        backbone, fold, stage, loss_type, out_dim = self.get_info_from_weight(weight_path)
        loss_config = decode_config(encode_config(loss_type=loss_type))
        weight_path = os.path.join(self.weight_dir, weight_path)
        if not os.path.exists(weight_path):
            print(f"Existing weights are {os.listdir(self.weight_dir)}")
            raise FileNotFoundError(f'{weight_path} does not exist')

        print(f'Loading model {backbone} - fold {fold} - stage {stage} - loss {loss_type}, dim {out_dim}')
        if backbone == 'resnest50':
            model = Resnest50(out_dim=out_dim, pretrained=False, loss_config=loss_config, args=self.args)
        else:
            model = Model(backbone, out_dim=out_dim, pretrained=False, loss_config=loss_config, args=self.args)
        model = model.cuda()
        checkpoint = torch.load(weight_path, map_location='cuda:0')
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        return model

    def load_models(self):
        models = []
        for weight_path in self.weight_list:
            model = self.load_effnets(weight_path)
            model.eval()
            models.append(model)

        return models

    def forward(self, img):
        results = []
        for model in self.models:
            if self.tta:
                tta_preds = []
                for trans_img in self.get_TTA(img):
                    feat, _ = model(trans_img)
                    tta_preds.append(feat)

                mean_pred = torch.mean(torch.stack(tta_preds), dim=0)
                results.append(l2_norm(mean_pred))
            else:
                feat, _ = model(img)
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
    def get_info_from_weight(path):
        """
        Inputs form: {backbone}_fold{fold}_stage{stage}_{loss_type}_outdim{outdim}.pth
        e.g: tf_efficientnet_b1_ns_fold0_stage1_cos_outdim11014.pth
        """
        def reformat(value):
            try:
                return int(value)
            except:
                # if contains number, get number
                numbers = re.findall(r'[0-9]+', value)
                if numbers:
                    return int(numbers[0])
                return value

        # Get backbone first
        backbone, parts = path.split('_fold')
        fold, stage, loss_type, outdim = [reformat(v) for v in parts.split("_")]

        return backbone, fold, stage, loss_type, outdim


def inference(model, test_loader, tqdm=tqdm, normalize=False):
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

            if normalize:
                feat = l2_norm(feat)
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


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

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
