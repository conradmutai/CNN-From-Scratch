import numpy as np

from .activations import relu

class Conv2D:
    def __init__(self, weight, bias, pad=1):
        self.weight = weight
        self.bias = bias
        self.pad = pad

    def forward(self, image):
        if self.pad != 0:
            image = np.pad(
                image,
                pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode='constant'
            )

        batch_size, image_height_in, image_width_in = image.shape[0], image.shape[2], image.shape[3]
        channels_out, channels_in, kernel_height, kernel_width = self.weight.shape

        image_height_out = int(1 + image_height_in - kernel_height)
        image_width_out = int(1 + image_width_in - kernel_width)

        out = np.zeros(shape=(batch_size, channels_out, image_height_out, image_width_out), dtype=np.float32)

        for y in range(image_height_out):
            for x in range(image_width_out):
                for y_kernel in range(kernel_height):
                    for x_kernel in range(kernel_width):
                        this_pixel = image[0, 0, y + y_kernel, x + x_kernel]
                        this_weight = self.weight[0, 0, x_kernel, y_kernel]

                        out[0, 0, y, x] += this_weight * this_pixel

                out[0, 0, y, x] = relu(out[0, 0, y, x] + self.bias)

        return out

    def back(self, image):
        pass  # temporary hold


class MaxPool:
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
        self.input = None
        self.weight = weight
        self.bias = bias

    def forward(self, image):
        self.input = image
        return relu(image @ self.weight + self.bias)

    def backward(self):
        pass  # temp hold
