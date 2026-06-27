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
    def __init__(self, weights, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = weights
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0

    def step(self, weight_grad):
        self.t += 1

        self.m = self.beta1 * self.m + ((1 - self.beta1) * weight_grad)
        self.v = self.beta2 * self.v + ((1 - self.beta2) * weight_grad**2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        self.weights = self.weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return self.weights
