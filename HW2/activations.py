import numpy as np


# This file implements various activation functions that can be used in a layer in a well structured way
# In all of the activations, "forward" states the forward pass and "backward" states the backpropogation on that activation

def sigmoid(z, stage="forward"):
    forward = 1 / (1 + np.exp(-z))
    if stage == "forward":
        return forward
    elif stage == "backward":
        return forward * (1 - forward)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")


def relu(z, stage="forward"):
    if stage == "forward":
        return np.maximum(0.0, z)
    elif stage == "backward":
        return np.where(z >= 0, 1, 0)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")


def leaky_relu(z, stage="forward", negative_slope=0.01):
    if stage == "forward":
        return np.maximum(negative_slope * z, z)
    elif stage == "backward":
        return np.where(z >= 0, 1, negative_slope)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")


def tanh(z, stage="forward"):
    forward = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    if stage == "forward":
        return forward
    elif stage == "backward":
        return 1 - (forward ** 2)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")

