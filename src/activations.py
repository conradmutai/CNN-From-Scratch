import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z, axis=-1):
    e_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e_z / np.sum(e_z, axis=axis, keepdims=True)


def sigmoid(z):
    sig_z = 1 / (1 + np.exp(-z))
    return sig_z