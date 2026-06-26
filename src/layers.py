import numpy as np

from .activations import relu, softmax

class Conv2D:
    def __init__(self, weight, bias, image, weight_optimizer, bias_optimizer, pad=1):
        self.weight = weight
        self.bias = bias
        self.input = image
        self.weight_optimizer = weight_optimizer
        self.bias_optimizer = bias_optimizer
        self.weight_grad = None
        self.bias_grad = None
        self.pad = pad

    def forward(self, image, stride=1):
        # padding the forward pass
        if self.pad != 0:
            image = np.pad(
                image,
                pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode='constant'
            )

        # assigns the size of elements in the image and weight matrices to the valid variables
        batch_size, image_height_in, image_width_in = image.shape[0], image.shape[2], image.shape[3]
        channels_out, channels_in, kernel_height, kernel_width = self.weight.shape

        # calculates the appropriate dimensions of the image out
        image_height_out = np.floor(1 + (image_height_in - kernel_height) / stride).astype(int)
        image_width_out = np.floor(1 + (image_width_in - kernel_width) / stride).astype(int)

        # creates empty out matrix for the conv2D
        out = np.zeros(shape=(batch_size, channels_out, image_height_out, image_width_out), dtype=np.float32)

        # iterates through all the elements of the image matrix to calculate the sum of the pixels multiplied by weight
        for y_kernel in range(kernel_height):
            for x_kernel in range(kernel_width):
                # slice of the image aligned with the output grid, for this kernel offset
                image_slice = image[
                              :, :,
                              y_kernel: y_kernel + image_height_out * stride: stride,
                              x_kernel: x_kernel + image_width_out * stride: stride
                              ]  # shape: (batch, channels_in, height_out, width_out)

                weight_slice = self.weight[:, :, y_kernel, x_kernel]  # shape: (channels_out, channels_in)

                # contract over channels_in, broadcast over batch/spatial
                out += np.einsum('bcyx,fc->bfyx', image_slice, weight_slice)

        self.input = image

        return out

    def backward(self, gradient):
        filter_index, channels_in, m, n = self.weight.shape
        batch_size, channels_out, height_out, width_out = gradient.shape

        b_s, ch_in, img_height, img_width = self.input.shape

        weight_grad = np.zeros(shape=(filter_index, channels_in, m, n), dtype=np.float32)
        input_grad = np.zeros(shape=(b_s, ch_in, img_height, img_width), dtype=np.float32)
        bias_grad = np.sum(gradient, axis=(0, 2, 3))

        stride = 1  # update this if you ever pass a non-default stride into forward

        for i in range(m):
            for j in range(n):
                # slice of self.input aligned with the output grid, for this kernel offset
                input_slice = self.input[
                              :, :,
                              i: i + height_out * stride: stride,
                              j: j + width_out * stride: stride
                              ]  # shape: (batch, channels_in, height_out, width_out)

                # weight_grad[:, :, i, j]: contract over batch + spatial, keep channels_out and channels_in separate
                weight_grad[:, :, i, j] = np.einsum('bcyx,bfyx->fc', input_slice, gradient)

                # input_grad contribution for this kernel offset: contract over channels_out, scatter into the right spatial slice
                input_grad[
                    :, :,
                    i: i + height_out * stride: stride,
                    j: j + width_out * stride: stride
                ] += np.einsum('bfyx,fc->bcyx', gradient, self.weight[:, :, i, j])

        if self.pad != 0:
            input_grad = input_grad[:, :, self.pad:-self.pad, self.pad:-self.pad]

        self.weight_grad = weight_grad
        self.bias_grad = bias_grad

        return weight_grad, bias_grad, input_grad

    def update(self):
        self.weight = self.weight_optimizer.step(self.weight_grad)
        self.bias = self.bias_optimizer.step(self.bias_grad)

    def relu(self):
        pass


class MaxPool:
    def __init__(self):
        self.input_shape = None
        self.mask = None

    def forward(self, x_in):
        # assigns the dimension shapes to the appropriate variables
        batch_size, channels, height_in, width_in = x_in.shape

        # calculates the height out and width out
        height_out = height_in // 2
        width_out = width_in // 2

        # creates a null matrix to store the values post maxpooling
        x_out = np.zeros((batch_size, channels, height_out, width_out))

        # updates the mask and input shape
        self.mask = np.zeros_like(x_in)
        self.input_shape = x_in.shape

        # iterates through all the values in the input matrix
        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_out):  # height
                    for j in range(width_out):  # width
                        patch = x_in[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2]  # gets the patch from the image
                        max_val = np.max(patch)  # gets the max from each quarter

                        x_out[b, c, i, j] = max_val  # assigns the max value to the out matrix

                        # record where the max came from, within this patch
                        patch_mask = (patch == max_val)
                        self.mask[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] = patch_mask

        return x_out

    def backward(self, loss):
        # creates a loss out matrix and once again assigns the sizes of the loss shape to the appropriate variables
        loss_out = np.zeros(self.input_shape)
        batch_size, channels, height_out, width_out = loss.shape

        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_out):
                    for j in range(width_out):
                        loss_out[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] += (
                                self.mask[b, c, i * 2:i * 2 + 2, j * 2:j * 2 + 2] * loss[b, c, i, j]
                        )  # sets the loss out matrix max from the mask values obtained

        return loss_out

    def update(self):
        pass

    def relu_act(self):
        relu()


class Flatten:
    def __init__(self):
        self.input_shape = None

    # created to flatten a matrix into a 1 dimensional shape
    def forward(self, image):
        self.input_shape = image.shape
        batch_size = image.shape[0]

        return image.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def update(self):
        pass


class FullyConnected:
    def __init__(self, weight, bias, weight_optimizer, bias_optimizer):
        self.grad_bias = None
        self.grad_weight = None
        self.input = None
        self.weight = weight
        self.bias = bias
        self.weight_optimizer = weight_optimizer
        self.bias_optimizer = bias_optimizer

    def forward(self, image):
        self.input = image
        return image @ self.weight + self.bias

    def backward(self, grad_output):
        # grad_output: dL/d(output of this layer), shape = (batch, out_features)

        # Gradients w.r.t. parameters
        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)

        # Gradient to pass to the previous layer
        grad_input = grad_output @ self.weight.T
        return grad_input

    def update(self):
        self.weight = self.weight_optimizer.step(self.grad_weight)
        self.bias = self.bias_optimizer.step(self.grad_bias)


class BatchNorm2D:
    def __init__(self, epsilon, beta, gamma, sigma, gamma_grad, beta_grad, gamma_optimizer, beta_optimizer):
        self.s_h = None
        self.x_hat = None
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.gamma_grad = gamma_grad
        self.beta_grad = beta_grad
        self.gamma_optimizer = gamma_optimizer
        self.beta_optimizer = beta_optimizer

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

                        h[b, c, y, x] = self.gamma[c] * x_hat_i + self.beta[c]

        self.x_hat = x_hat
        self.s_h = s_h_all

        return h

    def backward(self, gradient):
        # assigns these variables to the size of batches, number of channels, height of the image, and width of the image
        batch_size, channels_in, image_height, image_width = gradient.shape

        # calculates the number of elements
        m = batch_size * image_height * image_width

        beta_grad = np.sum(gradient, axis=(0, 2, 3))  # calculates the beta gradient by summing gradient
        gamma_grad = np.sum(gradient*self.x_hat, axis=(0, 2, 3))  # calculates the sum of the gradient times the hidden unit output
        x_hat_grad = gradient * self.gamma.reshape(1, channels_in, 1, 1)  # calculates the gradient of the hidden unit outputs
        sum_x_hat_grad = np.sum(x_hat_grad, axis=(0, 2, 3))  # sums the hidden unit output gradients
        sum_x_hat_grad_x_hat = np.sum(x_hat_grad * self.x_hat, axis=(0, 2, 3))  # multiplies the sum of hidden unit gradients by the gradient

        x_out = 1/(m*self.s_h.reshape(1, channels_in, 1, 1)) * (m * x_hat_grad - sum_x_hat_grad.reshape(1, channels_in, 1, 1) - (self.x_hat * sum_x_hat_grad_x_hat.reshape(1, channels_in, 1, 1)))

        self.gamma_grad = gamma_grad
        self.beta_grad = beta_grad

        return x_out

    def update(self):
        self.gamma = self.gamma_optimizer.step(self.gamma_grad)
        self.beta = self.beta_optimizer.step(self.beta_grad)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return relu(self.input)

    def backward(self, grad_output):
        grad_input = grad_output * (1 if self.input > 0 else 0)
        return grad_input

    def update(self):
        pass