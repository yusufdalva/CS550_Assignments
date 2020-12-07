import numpy as np


def bce(y, y_pred, stage="forward"):
    assert y_pred.shape == y.shape
    if stage == "forward":
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    elif stage == "backward":
        return y / y_pred - (1 - y) / (1 - y_pred)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")


def mse(y, y_pred, stage="forward"):
    assert y_pred.shape == y.shape
    if stage == "forward":
        return np.square(y - y_pred)
    elif stage == "backward":
        return -2 * (y - y_pred)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")


def sum_of_squares(y, y_pred, stage="forward"):
    assert y_pred.shape == y.shape
    if stage == "forward":
        return np.square(y, y_pred)
    elif stage == "backward":
        return -2 * (y - y_pred)
    else:
        raise ValueError("Invalid function stage, should be either \"forward\" or \"backward\"")
