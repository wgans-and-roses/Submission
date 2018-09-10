import torch
from models import *

opt = {'pretrained': True}
net = Alexnet(opt)

print(net)
