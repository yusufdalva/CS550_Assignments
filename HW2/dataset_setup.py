import os
import shutil
import wget
import numpy as np

DATA_FILE_NAMES = ["train1", "test1", "train2", "test2"]
BASE_URL = "http://www.cs.bilkent.edu.tr/~gunduz/teaching/cs550/documents"
FILE_URLS = [os.path.join(BASE_URL, file_name) for file_name in DATA_FILE_NAMES]


def download_data(file_names, file_urls):
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "data")):
        shutil.rmtree(os.path.join(cwd, "data"))
    folder_path = os.path.join(cwd, "data")
    os.mkdir(folder_path)
    file_paths = []
    for file_idx in range(len(file_names)):
        file_path = os.path.join(folder_path, file_names[file_idx] + ".txt")
        file_paths.append(file_path)
        wget.download(file_urls[file_idx], file_path)
    print("\nDOWNLOAD COMPLETED")
    return file_paths


class Dataset:

    def __init__(self, train_data_path, test_data_path):
        self.train_data = self.read_data(train_data_path)
        self.test_data = self.read_data(test_data_path)

    @staticmethod
    def read_data(data_path):
        train_file = open(data_path, "r")
        samples = train_file.readlines()
        sample_data = [list(map(float, sample.split())) for sample in samples]
        return np.array(sample_data)
