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
        pass # temporary hold


class maxPool:
    def __init__(self):

    def forward(self):



def flatten( ):


def fullyConnected( ):
