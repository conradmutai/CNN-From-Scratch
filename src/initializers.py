import numpy as np


# xavier initialization
def xavier_init(weights, mode):  # number of neuron inputs and number of neuron outputs
    channels_out, channels_in, kernel_h, kernel_w = weights.shape

    fan_in = channels_in * kernel_h * kernel_w
    fan_out = channels_out * kernel_h * kernel_w

    if mode == 'Uniform':
        x = np.sqrt(6/(fan_in + fan_out))
        w_out = np.random.uniform(low=-x, high=x, size=(channels_out, channels_in, kernel_h, kernel_w))  # generates a random weight matrix
    else:  # mode is normalized
        sigma = np.sqrt(2/(fan_in + fan_out))
        w_out = np.random.normal(loc=0.0, scale=sigma, size=(channels_out, channels_in, kernel_h, kernel_w))

    return w_out


def he_initialization(weights, distribution, mode):
    channels_out, channels_in, kernel_h, kernel_w = weights.shape

    fan_in = channels_in * kernel_h * kernel_w
    fan_out = channels_out * kernel_h * kernel_w

    if distribution == 'Uniform':
        if mode == 'Fan-in':
            bound = np.sqrt(6/fan_in)
            w_out = np.random.uniform(low=-bound, high=bound, size=(channels_out, channels_in, kernel_h, kernel_w))
        else:  # fan-out
            bound = np.sqrt(6/fan_out)
            w_out = np.random.uniform(low=-bound, high=bound, size=(channels_out, channels_in, kernel_h, kernel_w))
    else:  # normal distribution
        if mode == 'Fan-in':
            bound = np.sqrt(2/fan_in)
            w_out = np.random.normal(loc=0.0, scale=bound, size=(channels_out, channels_in, kernel_h, kernel_w))
        else:  # fan out
            bound = np.sqrt(2/fan_out)
            w_out = np.random.normal(loc=0.0, scale=bound, size=(channels_out, channels_in, kernel_h, kernel_w))

    return w_out