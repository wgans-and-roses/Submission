import preprocessing_module as preproc
from torch.utils.data import DataLoader
import torchvision as tv
from custom_transforms import Crop
from util.parser import Parser
from util.io import *
from models import *
from os.path import join
import torch.nn as nn
from util.metrics import *
from itertools import cycle
import copy

args = \
[
[('--lr',), {'type': float, 'default': 5E-5, 'help': 'Learning rate'}],
[('--lrs',), {'type': int, 'default': [2, 4, 6, 8], 'nargs': '+', 'help': 'Learning rate schedule'}],
[('--lrd',), {'type': float, 'default': 0.2, 'help': 'Learning rate decay'}],
[('--l2',), {'type': float, 'default': 0.5E-5, 'help': 'L2 regularization'}],
[('--train_path',), {'type': str, 'default': '/mnt/DATA/beantech_contestAI/Dataset2/', 'help': 'Location of the dataset'}],
[('--validation_path',), {'type': str, 'default': '/mnt/DATA/beantech_contestAI/Dataset1/', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '.', 'help': 'Results path'}],
[('--batch_size', '-bs'), {'type': int, 'default': 64, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 10, 'help': 'Number of epochs'}]
]

argparser = Parser("Beantech challenge")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

W = 1280
H = 180

path_training_ok = join(opt['train_path'],'campioni OK')
path_training_ko = join(opt['train_path'],'campioni KO')
path_validation_ok = join(opt['validation_path'],'campioni OK')
path_validation_ko = join(opt['validation_path'],'campioni KO')

num_epochs = 1
dataset_name = 'albedo'

transform = tv.transforms.Compose([Crop(H), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=[0.5],
                                 std=[1.0])])
(training_set_ok,), (training_set_ko,) = preproc.build_datasets(path_training_ok, path_training_ko, dataset_name, transform, split=True)

training_loader_ok = DataLoader(training_set_ok, batch_size=int(opt['batch_size']/2), shuffle=True)
training_loader_ko = DataLoader(training_set_ko, batch_size=int(opt['batch_size']/2), shuffle=True)

validation_set, = preproc.build_datasets(path_validation_ok, path_validation_ko, dataset_name, transform)

val, test = preproc.split(validation_set, 0.7, random_state=56)
val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

val_data = next(iter(val_loader))
del val_loader, val
test_data = next(iter(test_loader))
del test_loader, test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval(model, loss_fcn, data):
    model.eval()
    X = data['image'].float().to(device)
    Y = data['image_label'].float().view(-1,1).to(device)
    X = X.expand(-1, 3, -1, -1)
    with torch.no_grad():
        out = model(X )
        y_hat = torch.round(out)
        acc = binary_accuracy(Y, y_hat)
        f1_score = f1(Y, y_hat)
        loss = loss_fcn(out, Y)
    model.train()
    return acc.tolist(), loss.tolist(), f1_score


model = Alexnet()
model = model.to(device)

for par in model.features.parameters():
    par.requires_grad = False
param = model.classifier.parameters()

optimizer = torch.optim.Adam(param, lr=opt['lr'], weight_decay=opt['l2'])

idx = 0
loss_win = []; acc_win = []; f1_win = []
vis_period = 5
last_loss = 0; last_val_loss = 0; last_val_acc = 0; max_f1 = 0; last_val_f1 = 0
loss_fcn = nn.BCELoss()
best_model = []

for epoch in range(1, opt['epochs']+1):
    print("Epoch: " + str(epoch) + "/" + str(opt['epochs']))

    if epoch in opt['lrs']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * opt['lrd']

    for batch_ok, batch_ko in zip(training_loader_ok, cycle(training_loader_ko)):
        X = torch.cat((batch_ok['image'].float().to(device),
                       batch_ko['image'].float().to(device)))
        Y = torch.cat((batch_ok['image_label'].float().view(-1,1).to(device),
                      batch_ko['image_label'].float().view(-1,1).to(device)))

        X = X.expand(-1, 3, -1, -1)

        model.zero_grad()
        out = model(X)
        loss = loss_fcn(out, Y)
        loss.backward()
        optimizer.step()

        val_acc, val_loss, val_f1 = eval(model, loss_fcn, val_data)
        if val_f1 > max_f1:
            max_f1 = val_f1
            best_model = copy.deepcopy(model)
            save_checkpoint(best_model, epoch, optimizer, opt, join(opt['save_path'], 'best_model.m'))
            print('Best validation F1: ' + str(val_f1))
        idx += 1

acc, loss, f1_score = eval(best_model.to(device), loss_fcn, test_data)
print("Test F1: " + str(f1_score))