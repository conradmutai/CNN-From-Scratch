import numpy as np


def cross_entropy_loss(actual_y, predicted_y):
    epsilon = 1e-12
    predicted_y = np.clip(predicted_y, epsilon, 1 - epsilon)
    return -np.sum(actual_y * np.log(predicted_y))


def gradient_loss(weight, d_loss, d_weight, learning_rate):
    return weight - learning_rate * (d_loss / d_weight)
