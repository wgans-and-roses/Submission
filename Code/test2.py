import preprocessing_module as preproc
from torch.utils.data import DataLoader
from visdom import Visdom
from util.visual import VisualManager
import torchvision as tv
from custom_transforms import Crop
from util.parser import Parser
from util.io import *
from models import *
from os.path import join
import torch.nn as nn
import time
from util.metrics import *
from itertools import cycle
import torchvision.models as models

args = \
[
[('--model', '-m'), {'type': str, 'default': 'lenet', 'help': 'Used Model lenet | allcnn |'}],
[('--lr',), {'type': float, 'default': 0.001, 'help': 'Learning rate'}],
[('--lrs',), {'type': int, 'default': [30, 60, 90], 'nargs': '+', 'help': 'Learning rate schedule'}],
[('--lrd',), {'type': float, 'default': 0.1, 'help': 'Learning rate decay'}],
[('--l2',), {'type': float, 'default': 0.0, 'help': 'L2 regularization'}],
[('-d',), {'type': float, 'default': 0.0, 'help': 'Dropout probability'}],
[('--dataset',), {'type': str, 'default': 'mnist', 'help': 'Dataset to use'}],
[('--root',), {'type': str, 'default': '/mnt/DATA/TorchData', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '/mnt/DATA/ProjectsResults/contestai', 'help': 'Results path'}],
[('--batch_size', '-bs'), {'type': int, 'default': 64, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 120, 'help': 'Number of epochs'}],
[('--log_period', '-lp'), {'type': int, 'default': 20, 'help': 'Logging period in number of epochs'}],
#[('--optimizer', '-opt'), {'type': str, 'default': 'adam', 'help': 'Optimizer to use'}],
[('--pretrained', '-pr'), {'type': int, 'default': True, 'help': 'Use the pretrained model?'}]
]

argparser = Parser("Beantech challenge")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

dirname = build_dirname(opt, ('lr', 'batch_size'))
savepath = make_save_directory(opt, dirname)

vis = Visdom(port=8098)
vm = VisualManager(vis, 'contestai')

W = 1280
H = 180

path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

num_epochs = 1
dataset_name = 'albedo'

tranform = tv.transforms.Compose([Crop(H), tv.transforms.ToTensor()])
# (training_set_ok,), (training_set_ko,) = preproc.build_datasets(path_training_ok, path_training_ko, dataset_name, tranform, split=True)
# training_loader_ok = DataLoader(training_set_ok, batch_size=int(opt['batch_size']/2), shuffle=True)
# training_loader_ko = DataLoader(training_set_ko, batch_size=int(opt['batch_size']/2), shuffle=True)

validation_set, = preproc.build_datasets(path_validation_ok, path_validation_ko, dataset_name, tranform)

val, test = preproc.split(validation_set, 0.7, random_state=55)
val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

val_data = next(iter(val_loader))
del val_loader, val
# test_data = next(iter(test_loader))
# del test_loader, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%

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
            nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,10)),
            View(1),
            nn.Sigmoid()
        )

        self.Nf = num_parameters(self.features)
        self.Nc =  num_parameters(self.classifier)
        s = '[%s] Features parameters: %d, Classifier Parameters: %d' % (self.name, self.Nf, self.Nc)
        print(s)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)


def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

def eval(model, loss_fcn, data):
    model.eval()
    X = data['image'].float().to(device)
    Y = data['image_label'].float().view(-1,1).to(device)
    out = model(X)
    y_hat = torch.round(out)
    acc = binary_accuracy(Y, y_hat)
    loss = loss_fcn(out, Y)
    model.train()
    return acc.clone().tolist(), loss.clone().tolist()


model = AlexnetAllcnn(opt)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['l2'])

# optimizer = torch.optim.SGD(model.parameters(),
#                          weight_decay=opt['l2'],
#                          lr=opt['lr'],
#                          momentum=0.9,
#                          nesterov=True)

X = val_data['image'].float().to(device)
X = X.expand(-1, 3, -1, -1)
out = model(X)
