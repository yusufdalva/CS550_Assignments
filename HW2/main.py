from dataset_setup import *
from ann import *

DATA_FILE_NAMES = ["train1", "test1", "train2", "test2"]
BASE_URL = "http://www.cs.bilkent.edu.tr/~gunduz/teaching/cs550/documents"
FILE_URLS = [os.path.join(BASE_URL, file_name) for file_name in DATA_FILE_NAMES]

file_paths = download_data(DATA_FILE_NAMES, FILE_URLS)

dataset_1 = Dataset(file_paths[0], file_paths[1])
dataset_2 = Dataset(file_paths[2], file_paths[3])

network_1 = ANN(input_dim=1, weight_range=0.01, hidden_layer_enabled=False)

network_2 = ANN(input_dim=1, weight_range=0.001, hidden_layer_enabled=True, hidden_units=20, activation="sigmoid")

normalized_train, normalized_test = dataset_1.normalize_data()
network_2.fit(normalized_train, 10000, learning_rate=0.005, update="sgd", momentum_enabled=True, alpha=0.1)
