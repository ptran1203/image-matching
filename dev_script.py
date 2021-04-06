from models import *

class Effnet(nn.Module):
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
        super(Effnet, self).__init__()
        enet_type = enet_type.replace('-', '_')

        feat_dim = 512
        planes = self._get_global_dim(enet_type)
        self.enet = geffnet.create_model(enet_type,
            pretrained=pretrained, as_sequential=True)[:-4]
        # self.feat = nn.Linear(self.enet.classifier.in_features, feat_dim)
        self.swish = Swish_module()
        self.arc = ArcMarginProduct_subcenter(feat_dim, out_dim)

        self.local_conv = nn.Conv2d(planes, feat_dim, 1)
        self.local_bn = nn.BatchNorm2d(feat_dim)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(planes)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)

        # feature = F.adaptive_avg_pool2d(x, 1)
        global_feat = F.avg_pool2d(x, x.size()[2:])
        global_feat = global_feat.view(global_feat.size()[0], -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        # local feat
        local_feat = torch.mean(x, [2, 3], keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)
        logits_m = self.arc(local_feat)

        return global_feat, local_feat, logits_m

model = Effnet('tf_efficientnet_b3_ns', 100)
feat = model(torch.stack([img, img]))
feat[0].shape, feat[1].shape, feat[2].shape