import pretrainedmodels
from torch import nn


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x12):

        x1, x2 = x12
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):  # TODO
        return self.embedding_net(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


class ClassificationNet(nn.Module):
    def __init__(self, backbone_name, n_classes, frozen_encoder=True):
        super().__init__()
        self.n_classes = n_classes

        model = pretrainedmodels.__dict__[backbone_name]()
        dim_feats = model.last_linear.in_features
        if frozen_encoder:
            for param in model.parameters():
                param.requires_grad = False

        model.last_linear = nn.Sequential(nn.PReLU(),
                                          nn.Linear(dim_feats, n_classes),
                                          nn.LogSoftmax())

        self.net = model
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class LargerClassificationNet(nn.Module):
    def __init__(self, backbone_name, n_classes, frozen_encoder=True):
        super().__init__()
        self.n_classes = n_classes

        model = pretrainedmodels.__dict__[backbone_name]()
        dim_feats = model.last_linear.in_features
        if frozen_encoder:
            for param in model.parameters():
                param.requires_grad = False

        model.last_linear = nn.Sequential(nn.PReLU(),
                                          nn.Linear(dim_feats, 256),
                                          nn.PReLU(),
                                          nn.Linear(256, n_classes),
                                          nn.LogSoftmax())
        model.last_linear.apply(weights_init)

        self.net = model
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)
