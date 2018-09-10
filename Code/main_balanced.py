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
from util.metrics import *
from itertools import cycle
import copy
from torchnet.meter import *
args = \
[
[('--model', '-m'), {'type': str, 'default': 'alexnet', 'help': 'Used Model lenet | allcnn |'}],
[('--lr',), {'type': float, 'default': 5E-5, 'help': 'Learning rate'}],
[('--lrs',), {'type': int, 'default': [2, 4, 6, 8], 'nargs': '+', 'help': 'Learning rate schedule'}],
[('--lrd',), {'type': float, 'default': 0.2, 'help': 'Learning rate decay'}],
[('--l2',), {'type': float, 'default': 0.5E-5, 'help': 'L2 regularization'}],
[('--l1',), {'type': float, 'default': 0.5E-5, 'help': 'L1 regularization'}],
[('-d',), {'type': float, 'default': 0.5, 'help': 'Dropout probability'}],
[('--dataset',), {'type': str, 'default': 'mnist', 'help': 'Dataset to use'}],
[('--root',), {'type': str, 'default': '/mnt/DATA/TorchData', 'help': 'Location of the dataset'}],
[('--save_path', '-s'), {'type': str, 'default': '/mnt/DATA/ProjectsResults/contestai', 'help': 'Results path'}],
[('--batch_size', '-bs'), {'type': int, 'default': 64, 'help': 'Batch size'}],
[('--epochs', '-e'), {'type': int, 'default': 10, 'help': 'Number of epochs'}],
[('--log_period', '-lp'), {'type': int, 'default': 20, 'help': 'Logging period in number of epochs'}],
#[('--optimizer', '-opt'), {'type': str, 'default': 'adam', 'help': 'Optimizer to use'}],
[('--pretrained', '-pr'), {'type': int, 'default': True, 'help': 'Use the pretrained model?'}],
]

argparser = Parser("Beantech challenge")
argparser.add_arguments(args)
opt = argparser.get_dictionary()

dirname = build_dirname(opt, ('lr', 'batch_size', 'l2', 'lrs', 'd'))
savepath = make_save_directory(opt, dirname)

vis = Visdom(port=8098)
vm = VisualManager(vis, dirname)

W = 1280
H = 180

path_training_ok = '/mnt/DATA/beantech_contestAI/Dataset2/campioni OK'
path_training_ko = '/mnt/DATA/beantech_contestAI/Dataset2/campioni KO'
path_validation_ok = '/mnt/DATA/beantech_contestAI/Dataset1/campioni OK'
path_validation_ko = '/mnt/DATA/beantech_contestAI/Dataset1/campioni KO'

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

#%%
def eval(model, loss_fcn, data):
    model.eval()
    m = ConfusionMeter(k=2, normalized=True)
    X = data['image'].float().to(device)
    Y = data['image_label'].float().view(-1,1).to(device)
    if opt['model'] in ['alexnet', 'alexnetall']:
        X = X.expand(-1, 3, -1, -1)
    with torch.no_grad():
        out = model(X - avg)
        y_hat = torch.round(out)
        acc = binary_accuracy(Y, y_hat)
        f1_score = f1(Y, y_hat)
        m.add(y_hat.view(-1), Y.view(-1))
        confmat = m.value()
        loss = loss_fcn(out, Y)
    model.train()
    return acc.tolist(), loss.tolist(), confmat, f1_score


avg = torch.load(join(opt['save_path'], 'mean.m'))
avg = 0.0 # avg['mean'].to(device)

constructor = get_model(opt['model'])
model = constructor(opt)
model = model.to(device)

init = WightInitializer()

if opt['model'] in ['alexnet', 'alexnetall']:
    for par in model.features.parameters():
        par.requires_grad = False
    param = model.classifier.parameters()#torch.optim.Adam(model.classifier.parameters(), lr=opt['lr'], weight_decay=opt['l2'])
else:
    param = model.parameters()

optimizer = torch.optim.Adam(param, lr=opt['lr'], weight_decay=opt['l2'])
# optimizer = torch.optim.SGD(param,
#                          weight_decay=opt['l2'],
#                          lr=opt['lr'],
#                          momentum=0.99,
#                          nesterov=True)

idx = 0
loss_win = []; acc_win = []; f1_win = []
vis_period = 5
last_loss = 0; last_val_loss = 0; last_val_acc = 0; max_f1 = 0; last_val_f1 = 0
loss_fcn = nn.BCELoss()
best_model = []

#%%
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


        if opt['model'] in ['alexnet', 'alexnetall']:
            X = X.expand(-1, 3, -1, -1)

        model.zero_grad()
        out = model(X - avg)
        loss = loss_fcn(out, Y)
        loss.backward()
        optimizer.step()

        val_acc, val_loss, _, val_f1 = eval(model, loss_fcn, val_data)
        if val_f1 > max_f1:
            max_f1 = val_f1
            best_model = copy.deepcopy(model)
            save_checkpoint(best_model, epoch, optimizer, opt, join(savepath, 'best_model.m'))

        if idx%vis_period == 0:
            if idx != 0:
                x = torch.Tensor([idx - vis_period, idx])
                y = torch.Tensor([[last_loss, loss.clone().tolist()],
                                  [last_val_loss, val_loss]]).transpose(1,0)
                acc = torch.Tensor([last_val_acc, val_acc])
                f1_val = torch.Tensor([last_val_f1, val_f1])
                loss_win = vm.update(y, x, win=loss_win, title='Loss')
                f1_win = vm.update(f1_val, x, win=f1_win, title='F1')
                acc_win = vm.update(acc, x, win=acc_win, title='Validation Accuracy')
            last_loss = loss.clone().tolist()
            last_val_acc, last_val_loss, last_val_f1 = val_acc, val_loss, val_f1
        idx += 1

acc, loss, mat, f1_score = eval(best_model.to(device), loss_fcn, test_data)
op = {'xlabel': 'Predicted Class', 'ylabel': 'Real Class'}
vm.plot(np.array(mat), plot_type='heatmap', opts=op)
print(f1_score)