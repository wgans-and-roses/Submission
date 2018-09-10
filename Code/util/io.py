import h5py
import numpy as np
import torch
import json
import time
import os


def build_dirname(opt, keys):
    sub_dict = dict((k, opt[k]) for k in keys)
    th = time.asctime().split(' ')
    try:
        th.remove('')
    except:
        pass
    date_time = '[' + th[1] + '_' + th[2] + '_' + th[3] + ']'
    params = str(sub_dict).replace(" ", "")
    filename = date_time + '_par_' + params
    return filename


def make_save_directory(opt, dirname):
    if not os.path.exists(os.path.join(opt['save_path'], opt['model'])):
        os.mkdir(os.path.join(opt['save_path'], opt['model']))
    savepath = os.path.join(opt['save_path'], opt['model'], dirname)
    os.mkdir(savepath)
    return savepath


def save_hf5(variables, names, file_path):
    file = h5py.File(file_path, 'w')
    for var, name in zip(variables, names):
        file.create_dataset(name, data=var)
    file.close()


def load_hf5(file_path):
    file = h5py.File(file_path, 'r')
    for name in file.keys():
        data = file[name]
        globals()[name] = np.array(data)
    file.close()


def load_hf5_dictionary(file_path):
    file = h5py.File(file_path, 'r')
    dictionary = dict((k, np.array(file[k])) for k in file.keys())
    file.close()
    return dictionary


def save_checkpoint(model, epoch, optimizer, opt, file_path):
    torch.save(dict(
        opt=json.dumps(opt),
        model=model.state_dict(),
        e=epoch,
        t=optimizer.state['t']),
        file_path)