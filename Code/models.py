import torch.nn as nn
import torchvision.models as models


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)


class Alexnet(nn.Module):
    name = 'alexnet'
    def __init__(self):
        super().__init__()
        model = models.alexnet(pretrained=True)

        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=(4, 4), stride=(1, 4), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            View(32*3*10),
            nn.Linear(in_features=32*3*10, out_features=int(64)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=int(64), out_features=1),
            nn.Sigmoid()
        )

        self.Nf = num_parameters(self.features)
        self.Nc =  num_parameters(self.classifier)
        s = '[%s] Features parameters: %d, Classifier Parameters: %d' % (self.name, self.Nf, self.Nc)
        print(s)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)