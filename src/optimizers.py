import numpy as np


class SGD:  # stochastic gradient descent
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.lr = learning_rate

    def step(self, weight_grad):
        new_weight = self.weights - self.lr * weight_grad
        self.weights = new_weight
        return self.weights


class Adam:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.lr = learning_rate

    def step(self, weight_grad, epsilon=0.05):
        square_gradient = np.square(weight_grad)
        self.weights = self.weights - self.lr * (weight_grad/(square_gradient + epsilon))
        return self.weights
