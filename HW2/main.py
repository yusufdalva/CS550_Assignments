from dataset_setup import *
from network import *

DATA_FILE_NAMES = ["train1", "test1", "train2", "test2"]
BASE_URL = "http://www.cs.bilkent.edu.tr/~gunduz/teaching/cs550/documents"
FILE_URLS = [os.path.join(BASE_URL, file_name) for file_name in DATA_FILE_NAMES]

file_paths = download_data(DATA_FILE_NAMES, FILE_URLS)

dataset_1 = Dataset(file_paths[0], file_paths[1])
dataset_2 = Dataset(file_paths[2], file_paths[3])

layers = [(4, 1, 'relu'), (1, 4, 'sigmoid')]
network = Network(layers)

for sample_idx in range(dataset_1.train_data.shape[0]):
    y_pred = network.predict(dataset_1.train_data[sample_idx, :-1])
    print(y_pred)
