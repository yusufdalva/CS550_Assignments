import numpy as np
from activations import *
from losses import *

# In this file, the implementation of a hidden layer and a linear regressor is given
# Fot the purpose of this assignment, Linear Regressor is considered as a layer in the constructed ANN

# The ANN implemented uses a LinearRegressor layer and a nonlinear layer, these two instances only manage the weights
# All learning will be done on ANN instance


class LinearRegressor:

    def __init__(self, input_units, weight_range=0.01):
        self.weights = np.random.uniform(-weight_range, weight_range, (1, input_units))
        self.bias = np.random.uniform(-weight_range, weight_range, (1, 1))
        # The regressor has no activation functions but only applies
        # Linear discriminant function for prediction


class Layer:

    def __init__(self, input_units, hidden_units, activation, negative_slope=None, weight_range=0.01):
        self.weights = np.random.uniform(-weight_range, weight_range, (hidden_units, input_units))
        self.bias = np.random.uniform(-weight_range, weight_range, (hidden_units, 1))  # Each hidden unit has its own bias value

        # Negative slope attribute is reserved for LeakyReLU activation function
        # This value will only be used if it is not None and activation is LeakyReLU
        if activation == "relu":
            self.negative_slope = None
            self.activation = relu
        elif activation == "leaky_relu":
            self.negative_slope = negative_slope
            self.activation = leaky_relu
        elif activation == "sigmoid":
            self.negative_slope = None
            self.activation = sigmoid
        elif activation == "tanh":
            self.negative_slope = None
            self.activation = tanh
        else:
            raise ValueError("Invalid activation function: Should be one of ('relu', 'leaky_relu', 'sigmoid', 'tanh')")


class ANN:

    def __init__(self, input_dim, loss="mse", weight_range=0.01, hidden_layer_enabled=False, hidden_units=None, activation=None, negative_slope=None):
        if hidden_layer_enabled:
            if (not hidden_units) or (type(hidden_units) != int):
                raise ValueError("Invalid hidden unit value, should be an integer")
            hidden_layer = Layer(input_units=input_dim, hidden_units=hidden_units, activation=activation,
                                 negative_slope=negative_slope, weight_range=weight_range)
            linear_layer = LinearRegressor(input_units=hidden_units, weight_range=weight_range)
            self.layers = [hidden_layer, linear_layer]
        else:
            linear_layer = LinearRegressor(input_units=input_dim, weight_range=weight_range)
            self.layers = [linear_layer]
        if loss == "mse":
            self.loss = mse
        elif loss == "bce":
            self.loss = bce
        elif loss == "sum_of_squares":
            self.loss = sum_of_squares
        else:
            raise ValueError("Invalid loss function: Should be one of ('mse' (mean-squared error), 'bce' (binary cross-entropy), 'sum_of_squares')")

    def predict(self, samples):
        if len(self.layers) == 1:  # No hidden layers
            return (np.dot(self.layers[0].weights, samples.T) + self.layers[0].bias).flatten()
        else:  # Network with hidden layers
            assert type(self.layers[0]) == Layer
            hidden_z = np.dot(self.layers[0].weights, samples.T) + self.layers[0].bias
            hidden_out = self.layers[0].activation(hidden_z, stage="forward")
            linear_out = np.dot(self.layers[1].weights, hidden_out) + self.layers[1].bias
            return linear_out.flatten()

    def fit(self, train_data, epoch_count, learning_rate=0.01, update="batch", momentum_enabled=False, alpha=0.1, threshold=None):
        if len(self.layers) == 1:
            prev_grad = {"linear_w_grad": 0.0, "linear_b_grad": 0.0}
        else:
            prev_grad = {"linear_w_grad": 0.0, "linear_b_grad": 0.0, "hidden_w_grad": 0.0, "hidden_b_grad": 0.0}
        train_losses = []
        for epoch in range(epoch_count):
            np.random.shuffle(train_data)
            train_samples = train_data[:, :-1]
            train_output = train_data[:, -1]
            if update == "batch":
                y_pred, loss, cache = self.forward_pass(train_samples, train_output)
                train_losses.append(loss)
                if momentum_enabled:
                    prev_grad = self.backward_pass(train_output, y_pred, cache, learning_rate, momentum_enabled, prev_grad, alpha)
                else:
                    prev_grad = self.backward_pass(train_output, y_pred, cache, learning_rate)
                if threshold:
                    if loss < threshold:
                        return train_losses
            elif update == "sgd":
                epoch_loss = 0
                train_output = np.reshape(train_output, (len(train_output), 1))
                for sample_idx in range(len(train_samples)):
                    sample = np.reshape(train_samples[sample_idx, :], (1, train_samples.shape[1]))
                    y_pred, loss, cache = self.forward_pass(sample, train_output[sample_idx, :])
                    epoch_loss += loss
                    if momentum_enabled:
                        prev_grad = self.backward_pass(train_output[sample_idx], y_pred, cache, learning_rate, momentum_enabled, prev_grad, alpha)
                    else:
                        prev_grad = self.backward_pass(train_output[sample_idx], y_pred, cache, learning_rate)
                epoch_loss = epoch_loss / len(train_samples)
                if threshold:
                    if epoch_loss < threshold:
                        return train_losses
                train_losses.append(epoch_loss)
            else:
                raise ValueError("Invalid update method: Should be one of ('sgd' (Stochastic Gradient Descent), 'batch')")
        return train_losses

    def forward_pass(self, samples, labels):
        cache = []  # Will contain dictionaries storing the inputs to the layers - to be able to perform backpropogation
        labels = np.reshape(labels, (len(labels), 1))
        if len(self.layers) == 1:  # No hidden layers
            discriminant = np.dot(self.layers[0].weights, samples.T) + self.layers[0].bias
            y_pred = discriminant.T
            loss = np.asscalar(np.sum(self.loss(labels, discriminant.T, stage="forward"), axis=0) / len(samples))
            cache.append({"input": samples})
        else:  # Network with hidden layers
            discriminant = np.dot(self.layers[0].weights, samples.T) + self.layers[0].bias
            activation_value = self.layers[0].activation(discriminant, stage="forward")
            cache.append({"input": samples, "discriminant": discriminant})
            discriminant = np.dot(self.layers[1].weights, activation_value) + self.layers[1].bias
            y_pred = discriminant.T
            loss = np.asscalar(np.sum(self.loss(labels, discriminant.T, stage="forward"), axis=0) / len(samples))
            cache.append({"input": samples})
        return y_pred, loss, cache

    def backward_pass(self, labels, y_pred, cache, learning_rate=0.01, momentum_enabled=False, prev_grad=None, alpha=0.1):
        grad_cache = {}
        labels = np.reshape(labels, (len(labels), 1))
        loss_grad = self.loss(labels, y_pred, stage="backward")
        gradient = loss_grad
        if len(self.layers) == 1:  # No hidden layers
            grad_w = np.dot(gradient.T, cache[0]["input"]) / len(labels)
            grad_b = np.sum(gradient, axis=0) / len(labels)
            if momentum_enabled:
                grad_w = alpha * grad_w + (1 - alpha) * prev_grad["linear_w_grad"]
                grad_b = alpha * grad_b + (1 - alpha) * prev_grad["linear_b_grad"]
            self.layers[0].weights -= learning_rate * grad_w
            self.layers[0].bias -= learning_rate * grad_b
            grad_cache["linear_w_grad"] = grad_w
            grad_cache["linear_b_grad"] = grad_b
        else:  # Network with hidden layers
            # Linear Layer
            grad_w = np.dot(gradient.T, cache[1]["input"]) / len(labels)
            grad_b = np.sum(gradient, axis=0) / len(labels)
            gradient = gradient * self.layers[1].weights
            if momentum_enabled:
                grad_w = alpha * grad_w + (1 - alpha) * prev_grad["linear_w_grad"]
                grad_b = alpha * grad_b + (1 - alpha) * prev_grad["linear_b_grad"]
            self.layers[1].weights -= learning_rate * grad_w
            self.layers[1].bias -= learning_rate * grad_b
            grad_cache["linear_w_grad"] = grad_w
            grad_cache["linear_b_grad"] = grad_b
            # Hidden layer with non-linear activation
            activation_grad = self.layers[0].activation(cache[0]["discriminant"], stage="backward")
            grad_z = gradient.T * activation_grad
            grad_w = np.dot(grad_z, cache[0]["input"]) / len(labels)
            grad_b = np.sum(grad_z, axis=1, keepdims=True) / len(labels)
            if momentum_enabled:
                grad_w = alpha * grad_w + (1 - alpha) * prev_grad["hidden_w_grad"]
                grad_b = alpha * grad_b + (1 - alpha) * prev_grad["hidden_b_grad"]
            self.layers[0].weights -= learning_rate * grad_w
            self.layers[0].bias -= learning_rate * grad_b
            grad_cache["hidden_w_grad"] = grad_w
            grad_cache["hidden_b_grad"] = grad_b
        return grad_cache






