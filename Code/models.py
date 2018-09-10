import torch.nn as nn
import torchvision.models as models

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


def get_model(name):
    models = {}
    models['lenet'] = Lenet
    models['allcnn'] = Allcnn
    models['alexnet'] = Alexnet
    models['alexnetall'] = AlexnetAllcnn
    return models[name]


class Lenet(nn.Module):
    name = 'lenet'

    def __init__(self, opt, c1=20, c2=50, c3=100):
        super().__init__()

        opt['d'] = opt.get('d', 0.5)
        opt['l2'] = opt.get('l2', 0.0)

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d
        output_dim = 1

        def convbn(ci,co,ksz,psz):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz, padding=(1, 2), stride=(2, 4)),
                #bn2(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz, padding=1),
                nn.Dropout(opt['d']))

        self.m = nn.Sequential(
            convbn(1,c1,5,3),
            convbn(c1,c2,5,2),
            View(c2*8*14),
            nn.Linear(c2*8*14, c3),
            # #bn1(c3),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c3,output_dim),
            nn.Sigmoid())

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)

    def forward(self, x):
            return self.m(x)


class Allcnn(nn.Module):
    name = 'allcnn'

    def __init__(self, opt, c = 1, c1=48, c2=96): #c1=96, c2=192, microbn=False):
        super().__init__()

        if (not 'd' in opt) or opt['d'] < 0:
            if opt['augment']:
                opt['d'] = 0.0
            else:
                opt['d'] = 0.5

        output_dim = 1

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def convbn(ci,co,ksz,s,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                #bn2(co),
                #nn.ReLU(True)
                nn.LeakyReLU(True)
                )
        self.m = nn.Sequential(
            convbn(c,c1,3,2,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            #nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,1,3,2,1),
            nn.Dropout(opt['d']),
            #convbn(c2,output_dim,3,2,1),
            #convbn(c2,output_dim,1,1))
            #nn.AvgPool2d((6, 40)),
            View(12*80),
            nn.Linear(12*80, output_dim),
            nn.Sigmoid())

    def forward(self, x):
        return self.m(x)


class Alexnet(nn.Module):
    name = 'alexnet'
    def __init__(self, opt):
        super().__init__()
        model = models.alexnet(pretrained=opt['pretrained'])

        # relu = ['1', '4', '7', '9', '11']
        # for i in relu:
        #     model.features._modules[i] = nn.LeakyReLU(inplace=True)

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


class AlexnetAllcnn(nn.Module):
    name = 'alexnetall'
    def __init__(self, opt):
        super().__init__()
        model = models.alexnet(pretrained=opt['pretrained'])

        relu = ['1', '4', '7', '9', '11']
        for i in relu:
            model.features._modules[i] = nn.LeakyReLU(inplace=True)

        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(opt['d']),
            nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            View(4*10),
            nn.Linear(in_features=4*10, out_features=1),
            nn.Sigmoid()
        )

        self.Nf = num_parameters(self.features)
        self.Nc =  num_parameters(self.classifier)
        s = '[%s] Features parameters: %d, Classifier Parameters: %d' % (self.name, self.Nf, self.Nc)
        print(s)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class WightInitializer:
    def __init__(self, conv = (0, 0.02), bias = 0.01):
        self.bias = bias
        self.conv_std, self.conv_mean = conv

    def __call__(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, self.conv_mean, self.conv_std)
            m.bias.data.fill_(self.bias)