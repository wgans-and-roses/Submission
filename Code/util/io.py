import torch
import json


def save_checkpoint(model, epoch, optimizer, opt, file_path):
    torch.save(dict(
        opt=json.dumps(opt),
        model=model.state_dict(),
        e=epoch,
        t=optimizer.state['t']),
        file_path)