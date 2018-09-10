import preprocessing_module as preproc
from torch.utils.data import DataLoader
from visdom import Visdom
from util.visual import VisualManager
import torchvision as tv
from custom_transforms import Crop
from util.parser import Parser
from util.io import *
from models import Lenet, Allcnn
from os.path import join
import torch.nn as nn
import time
from util.metrics import *

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
]

argparser = Parser("Beantech challenge")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

dirname = build_dirname(opt, ('lr', 'batch_size'))
savepath = make_save_directory(opt, dirname)

vis = Visdom()
vm = VisualManager(vis, 'contestai')

W = 1280
H = 175

path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

num_epochs = 1
dataset_name = 'albedo'

tranform = tv.transforms.Compose([Crop(H), tv.transforms.ToTensor()])
training_set, = preproc.build_datasets(path_training_ok, path_training_ko, dataset_name, tranform)
training_loader = DataLoader(training_set, batch_size=opt['batch_size'], shuffle=True)

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

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

model = Lenet(opt)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['l2'])

# optimizer = torch.optim.SGD(model.parameters(),
#                          weight_decay=opt['l2'],
#                          lr=opt['lr'],
#                          momentum=0.9,
#                          nesterov=True)

idx = 0
loss_win = []; acc_win = []
vis_period = 30
last_loss = 0; last_val_loss = 0; last_val_acc = 0
loss_fcn = nn.BCELoss()
#%%
for epoch in range(1, opt['epochs']+1):
    print("Epoch: " + str(epoch) + "/" + str(opt['epochs']))

    if epoch in opt['lrs']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * opt['lrd']

    for batch in training_loader:
        X = batch['image'].float().to(device)
        Y = batch['image_label'].float().view(-1,1).to(device)

        model.zero_grad()
        out = model(X)
        loss = loss_fcn(out, Y)
        loss.backward()
        optimizer.step()

        if idx%vis_period == 0:
            val_acc, val_loss = eval(model, loss_fcn, val_data)
            if idx != 0:
                x = torch.Tensor([idx - vis_period, idx])
                y = torch.Tensor([[last_loss, loss.clone().tolist()],
                                  [last_val_loss, val_loss]]).transpose(1,0)
                acc = torch.Tensor([last_val_acc, val_acc])
                loss_win = vm.update(y, x, win=loss_win, title='Loss')
                acc_win = vm.update(acc, x, win=acc_win, title='Accuracy')
            last_loss = loss.clone().tolist()
            last_val_acc, last_val_loss = val_acc, val_loss
        idx += 1

    if epoch%opt['log_period'] == 0:
        save_checkpoint(model, epoch, optimizer, opt, join(savepath, 'model.m'))
