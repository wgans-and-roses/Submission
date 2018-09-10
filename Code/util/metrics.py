import torch
from sklearn.metrics import recall_score, precision_score

def binary_accuracy(y, y_hat):
    y = y.detach(); y_hat = y_hat.detach()
    y = y.view(-1); y_hat = y_hat.view(-1)  # this is to avoid problems if one vector is a column and the other is a row
    acc = torch.sum(y == y_hat).float()/y.size()[0]
    return acc


def precision(y, y_hat):
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    return precision_score(y, y_pred=y_hat)


def recall(y, y_hat):
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    return recall_score(y, y_pred=y_hat)


def f1(y, y_hat):
    return 2*(recall(y, y_hat) * precision(y, y_hat))/(recall(y, y_hat) + precision(y, y_hat))