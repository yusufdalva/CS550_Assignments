import numpy as np


class Layer:

    def __init__(self, hidden_units, input_units, activation, weight_range=1):
        self.weights = np.random.uniform(-weight_range, weight_range, (hidden_units, input_units))  # +1 for bias
        self. bias = np.random.uniform(-weight_range, weight_range, 1)
        if activation == "relu":
            self.activation = lambda z: np.maximum(0.0, z)
            self.backprop = lambda z: 1 if z >= 0 else 0
        elif activation == "sigmoid":
            self.activation = lambda z: 1 / (1 + np.exp(-z))
            self.backprop = lambda z: self.activation(z) * (1 - self.activation(z))
        elif activation == "leaky_relu":
            self.activation = lambda z: np.maximum(0.01 * z, z)
            self.backprop = lambda z: 1 if z >= 0 else 0.01
        elif activation == "tanh":
            self.activation = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
            self.backprop = lambda z: 1 - (self.activation(z) ** 2)
        else:
            raise ValueError("Invalid activation function input\nShould be one of ('relu', 'sigmoid', 'leaky_relu', 'tanh')")


class Network:
    # Takes list of tuples for layer hyper-parameters
    # Tuples are in the form (hidden_unit_count, input_unit_count, activation_name, weight_range (-w, w) - specifies w)
    def __init__(self, layer_params):
        self.layers = []
        for param in layer_params:
            if len(param) == 3:
                network_layer = Layer(param[0], param[1], param[2])
            else:
                network_layer = Layer(param[0], param[1], param[2], param[3])
            self.layers.append(network_layer)

    def predict(self, sample):
        if len(sample.shape) < 2:
            sample = np.reshape(sample, (sample.shape[0], 1))
        output = sample
        for layer in self.layers:
            z = np.dot(layer.weights, output) + layer.bias
            output = layer.activation(z)
        return output

    def fit(self, samples, labels):
        pass

    def evaluate(self, samples, labels):
        pass
