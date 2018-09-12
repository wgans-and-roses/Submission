from preprocessing_module import build_datasets
import torch
from custom_transforms import Crop
from torch.utils.data import DataLoader
import torchvision as tv
from models import Alexnet
import argparse


def build_string(name, value):
    name = name.split('_')
    name = name[0]
    string = "%s\t%d\n" % (name, value)
    return string


def evaluate_classification(path):
    W = 1280
    H = 180

    dataset_name = 'albedo'
    transform = tv.transforms.Compose([Crop(H), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=[0.5],
                                     std=[1.0])])
    dataset = build_datasets(path, dataset_name, transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_dict = torch.load('model.m', map_location='cpu')
    model = Alexnet()
    model.load_state_dict(loaded_dict['model'])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        data = next(iter(loader))
        X = data['image'].float().to(device)
        X = X.expand(-1, 3, -1, -1)
        pred = model(X).round().byte().view(-1)
    pred_str = [build_string(name, value) for name, value in zip(data['image_name'], pred.tolist())]
    with open("classification_results.txt", "a") as file:
        file.write(''.join(pred_str))
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model evaluation')
    parser.add_argument('path', type=str)
    args = vars(parser.parse_args())
    pred = evaluate_classification(args['path'])
