import numpy as np
import matplotlib.image as img
import os
import shutil
import wget


def reset_data_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


def download_data(file_url, name_to_save, reset=True):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data')
    if reset:
        reset_data_dir(data_dir)
    file_path = os.path.join(os.path.join(data_dir, name_to_save))
    wget.download(file_url, file_path)
    print('\nDOWNLOAD COMPLETED')
    return file_path


def load_data(file_path):
    data = img.imread(file_path)
    return np.array(data, dtype=np.float_)
