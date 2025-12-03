import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLayer(nn.Module):
    def __init__(self, config):
        super(FeatureLayer, self).__init__()
        if config.dataset in ["cifar10", "svhn"]:
            class_num = 10
        else:
            class_num = 100
        if config.model in ["resnet18", "preactresnet18"]:
            indim = 512
        else:
            indim = 640
        embed_dim = config.embed_dim
        self.linear = nn.Linear(indim, embed_dim)
        self.classifier = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        feature = self.linear(x)
        out = self.classifier(F.relu(feature))

        return feature, out