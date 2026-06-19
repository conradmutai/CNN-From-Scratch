import numpy as np


def relu(z):
    if z > 0:
        return z
    else:
        return 0


def softmax(z, axis=-1):
    e_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e_z / np.sum(e_z, axis=axis, keepdims=True)


def sigmoid(z):
    sig_z = 1 / (1 + np.exp(-1))
    return sig_z