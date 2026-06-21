import numpy as np

from .activations import relu

class Conv2D:
    def __init__(self, weight, bias, pad=1):
        self.weight = weight
        self.bias = bias
        self.pad = pad

    def forward(self, image, stride=1, pad=1):
        if self.pad != 0:
            image = np.pad(
                image,
                pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode='constant'
            )

        batch_size, image_height_in, image_width_in = image.shape[0], image.shape[2], image.shape[3]
        channels_out, channels_in, kernel_height, kernel_width = self.weight.shape

        image_height_out = np.floor(1 + (image_height_in - kernel_height) / stride).astype(int)
        image_width_out = np.floor(1 + (image_width_in - kernel_width) / stride).astype(int)

        out = np.zeros(shape=(batch_size, channels_out, image_height_out, image_width_out), dtype=np.float32)

        for y in range(image_height_out):
            for x in range(image_width_out):
                for y_kernel in range(kernel_height):
                    for x_kernel in range(kernel_width):
                        this_pixel = image[batch_size, channels_in, y * stride + kernel_height, x * stride + kernel_width]
                        this_weight = self.weight[channels_out, channels_in, kernel_height, kernel_width]

                        out[batch_size, channels_out, y, x] += np.sum(this_pixel * this_weight)

        return out

    def back(self, image):
        pass  # temporary hold


class MaxPool:
    def __init__(self):
        self.input_shape = None
        self.mask = None

    def forward(self, x_in):
        batch_size, channels, height_in, width_in = x_in.shape
        height_out = height_in // 2
        width_out = width_in // 2

        x_out = np.zeros((batch_size, channels, height_out, width_out))
        self.mask = np.zeros_like(x_in)
        self.input_shape = x_in.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_out):  # height
                    for j in range(width_out):  # width
                        patch = x_in[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                        max_val = np.max(patch)

                        x_out[b, c, i, j] = max_val

                        # record where the max came from, within this patch
                        patch_mask = (patch == max_val)
                        self.mask[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] = patch_mask

        return x_out

    def backward(self, loss):
        loss_out = np.zeros(self.input_shape)
        batch_size, channels, height_out, width_out = loss.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_out):
                    for j in range(width_out):
                        loss_out[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] += (
                                self.mask[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] * loss[b, c, i, j]
                        )

        return loss_out


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, image):
        self.input_shape = image.shape
        batch_size = image.shape[0]

        return image.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class FullyConnected:
    def __init__(self, weight, bias):
        self.grad_bias = None
        self.grad_weight = None
        self.input = None
        self.weight = weight
        self.bias = bias

    def forward(self, image):
        self.input = image
        return relu(image @ self.weight + self.bias)

    def backward(self, grad_output):
        # grad_output: dL/d(output of this layer), shape = (batch, out_features)

        # Backprop through ReLU
        relu_mask = (self.input @ self.weight + self.bias) > 0  # creates a boolean value
        grad_output = grad_output * relu_mask

        # Gradients w.r.t. parameters
        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)

        # Gradient to pass to the previous layer
        grad_input = grad_output @ self.weight.T
        return grad_input


class BatchNorm2D:
    def __init__(self, epsilon, beta, gamma, sigma, x_hat_i, s_h):
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.x_hat_i = x_hat_i
        self.s_h = s_h

    def forward(self, image):
        batch_size, channels_in, image_height, image_width = image.shape

        # creates an empty matrix for post batch normalization
        h = np.zeros(shape=(batch_size, channels_in, image_height, image_width), dtype=np.float32)
        # gets the number of elements
        m = batch_size * image_height * image_width

        # creates a place to hold the x_hat variables and std variables in a matrix after
        x_hat = np.zeros(shape=(batch_size, channels_in, image_height, image_width), dtype=np.float32)
        s_h_all = np.zeros(shape=(channels_in,), dtype=np.float32)

        for b in range(batch_size):
            for c in range(channels_in):
                m_h = 1 / m * np.sum(image[:, c])  # sums all the batches per a channel
                s_h = np.sqrt(1 / m * np.sum(np.square(image[:, c] - m_h)))  # gets the variance of the hidden layer

                s_h_all[c] = s_h  # stores the std of the hidden unit in the matrix

                for y in range(image_height):
                    for x in range(image_width):
                        x_hat_i = (image[b, c, y, x] - m_h) / (s_h + self.epsilon)
                        x_hat[b, c, y, x] = x_hat_i  # stores the x_hat inside its matrix store

                        h[b, c, y, x] = self.gamma[c] * x_hat_i + self.beta

        # UPDATES X_HAT_I AND S_H
        self.x_hat_i = x_hat
        self.s_h = s_h_all

        return h

    def backward(self, gradient):
        # assigns these variables to the size of batches, number of channels, height of the image, and width of the image
        batch_size, channels_in, image_height, image_width = gradient.shape

        # calculates the number of elements
        m = batch_size * image_height * image_width

        beta_grad = np.sum(gradient, axis=(0, 2, 3))  # calculates the beta gradient by summing gradient
        gamma_grad = np.sum(gradient*self.x_hat_i, axis=(0, 2, 3))  # calculates the sum of the gradient times the hidden unit output
        x_hat_grad = gradient * self.gamma.reshape(1, channels_in, 1, 1)  # calculates the gradient of the hidden unit outputs
        sum_x_hat_grad = np.sum(x_hat_grad, axis=(0, 2, 3))  # sums the hidden unit output gradients
        sum_x_hat_grad_xhat = np.sum(x_hat_grad * self.x_hat_i, axis=(0, 2, 3))  # multiplies the sum of hidden unit gradients by the gradient

        x_out = 1/(m*self.s_h) * (m * x_hat_grad - sum_x_hat_grad.reshape(1, channels_in, 1, 1) - (self.x_hat_i * sum_x_hat_grad_xhat.reshape(1, channels_in, 1, 1)))

        return x_out, gamma_grad, beta_grad


