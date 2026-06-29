import numpy as np

from .activations import relu


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
        self.stride = 1

    def forward(self, image, stride=1):
        self.stride = stride

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

        for y_kernel in range(kernel_height):
            for x_kernel in range(kernel_width):
                image_slice = image[
                              :, :,
                              y_kernel: y_kernel + image_height_out * stride: stride,
                              x_kernel: x_kernel + image_width_out * stride: stride
                              ]
                weight_slice = self.weight[:, :, y_kernel, x_kernel]
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

        stride = self.stride  # use the stride that was actually used in forward()

        for i in range(m):
            for j in range(n):
                input_slice = self.input[
                              :, :,
                              i: i + height_out * stride: stride,
                              j: j + width_out * stride: stride
                              ]

                weight_grad[:, :, i, j] = np.einsum('bcyx,bfyx->fc', input_slice, gradient)

                input_grad[
                    :, :,
                    i: i + height_out * stride: stride,
                    j: j + width_out * stride: stride
                ] += np.einsum('bfyx,fc->bcyx', gradient, self.weight[:, :, i, j])

        if self.pad != 0:
            input_grad = input_grad[:, :, self.pad:-self.pad, self.pad:-self.pad]

        self.weight_grad = clip_gradient(weight_grad)
        self.bias_grad = clip_gradient(bias_grad)

        return input_grad

    def update(self):
        self.weight = self.weight_optimizer.step(self.weight_grad)
        self.bias = self.bias_optimizer.step(self.bias_grad)


class MaxPool:
    def __init__(self):
        self.input_shape = None
        self.mask = None

    def forward(self, x_in):
        batch_size, channels, height_in, width_in = x_in.shape
        height_out = height_in // 2
        width_out = width_in // 2

        # crop in case height/width is odd (drop the leftover row/col)
        x_in = x_in[:, :, :height_out * 2, :width_out * 2]

        # reshape so each 2x2 window becomes its own pair of axes
        # shape: (batch, channels, height_out, 2, width_out, 2)
        reshaped = x_in.reshape(batch_size, channels, height_out, 2, width_out, 2)

        # max over the two pooling axes (3 and 5)
        x_out = reshaped.max(axis=(3, 5))

        # build the mask: which of the 4 positions in each window was the max
        # broadcast x_out back up to compare against the full window
        x_out_broadcast = x_out[:, :, :, None, :, None]
        mask = (reshaped == x_out_broadcast)

        self.mask = mask.reshape(x_in.shape)
        self.input_shape = (batch_size, channels, height_in, width_in)

        return x_out

    def backward(self, loss):
        batch_size, channels, height_out, width_out = loss.shape

        # broadcast loss back into the 2x2 windows using the saved mask
        loss_expanded = loss[:, :, :, None, :, None]
        mask_reshaped = self.mask.reshape(batch_size, channels, height_out, 2, width_out, 2)

        loss_out = (mask_reshaped * loss_expanded).reshape(
            batch_size, channels, height_out * 2, width_out * 2
        )

        # pad back to original shape if input had odd height/width
        full_h, full_w = self.input_shape[2], self.input_shape[3]
        if loss_out.shape[2] != full_h or loss_out.shape[3] != full_w:
            padded = np.zeros(self.input_shape, dtype=loss_out.dtype)
            padded[:, :, :loss_out.shape[2], :loss_out.shape[3]] = loss_out
            loss_out = padded

        return loss_out

    def update(self):
        pass


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
        self.grad_weight = clip_gradient(self.input.T @ grad_output)
        self.grad_bias = clip_gradient(np.sum(grad_output, axis=0))

        # Gradient to pass to the previous layer
        grad_input = grad_output @ self.weight.T
        return grad_input

    def update(self):
        self.weight = self.weight_optimizer.step(self.grad_weight)
        self.bias = self.bias_optimizer.step(self.grad_bias)


class BatchNorm2D:
    def __init__(self, epsilon, beta, gamma, gamma_optimizer, beta_optimizer, momentum=0.9):
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.gamma_optimizer = gamma_optimizer
        self.beta_optimizer = beta_optimizer
        self.momentum = momentum

        channels = gamma.shape[0]
        self.running_mean = np.zeros(channels, dtype=np.float32)
        self.running_var = np.ones(channels, dtype=np.float32)

        self.x_hat = None
        self.s_h = None
        self.training = True  # set to False at inference time

    def forward(self, image):
        batch_size, channels_in, image_height, image_width = image.shape

        if self.training:
            # mean/var per channel, computed across batch + spatial dims at once
            m_h = image.mean(axis=(0, 2, 3))                     # shape: (channels,)
            var_h = image.var(axis=(0, 2, 3))                    # shape: (channels,)
            s_h = np.sqrt(var_h)

            # update running stats for inference later
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * m_h
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var_h
        else:
            m_h = self.running_mean
            s_h = np.sqrt(self.running_var)

        # reshape stats to broadcast across (batch, channels, height, width)
        m_h_b = m_h.reshape(1, channels_in, 1, 1)
        s_h_b = s_h.reshape(1, channels_in, 1, 1)

        x_hat = (image - m_h_b) / (s_h_b + self.epsilon)
        h = self.gamma.reshape(1, channels_in, 1, 1) * x_hat + self.beta.reshape(1, channels_in, 1, 1)

        self.x_hat = x_hat
        self.s_h = s_h

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

        x_out = 1/(m*self.s_h.reshape(1, channels_in, 1, 1) + self.epsilon) * (m * x_hat_grad - sum_x_hat_grad.reshape(1, channels_in, 1, 1) - (self.x_hat * sum_x_hat_grad_x_hat.reshape(1, channels_in, 1, 1)))

        self.gamma_grad = clip_gradient(gamma_grad)
        self.beta_grad = clip_gradient(beta_grad)

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
        grad_input = grad_output * (self.input > 0)
        return grad_input

    def update(self):
        pass


# helper function to aid with gradient clipping
def clip_gradient(grad, max_norm=5.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm/norm)
    return grad

