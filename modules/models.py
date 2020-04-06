from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNetEncoder(torch.nn.Module):  # TODO
    embed_size = 512

    def __init__(self, pretrained=True, frozen=False):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)

        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    @staticmethod
    def from_siamese_ckpt(ckpt_path):
        encoder = ResNetEncoder(pretrained=False)
        siamese_model = SiameseNet(encoder)
        state_dict = torch.load(ckpt_path)['model_state_dict']
        siamese_model.load_state_dict(state_dict)

        return siamese_model.embedding_net


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


class ClassificationNet(nn.Module):
    def __init__(self, embed_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embed_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(embed_net.embed_size, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
