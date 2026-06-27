import numpy as np

from sklearn.datasets import fetch_openml

from .activations import softmax
from .layers import Conv2D, ReLU, MaxPool, Flatten, FullyConnected, BatchNorm2D
from .losses import cross_entropy_loss
from .optimizers import Adam, SGD
from .initializers import he_initialization, xavier_init
from .model import Model

# Convolutional layer 1
# Initializing parameters for the conv1
weight1 = xavier_init(np.zeros(shape=(16, 1, 3, 3), dtype=np.float32), "Uniform")  # this model uses xavier instead
bias1 = np.zeros(shape=(16,), dtype=np.float32)
weight1_optimizer = Adam(weight1)
bias1_optimizer = Adam(bias1)

# Initializing parameters for batchnorm1
gamma1 = np.ones(shape=(16,), dtype=np.float32)
beta1 = np.zeros(shape=(16,), dtype=np.float32)
gamma1_optimizer = Adam(gamma1)
beta1_optimizer = Adam(beta1)

conv1 = Conv2D(weight1, bias1, None, weight1_optimizer, bias1_optimizer)  # convolutional layer
batchnorm1 = BatchNorm2D(epsilon=1e-5, beta=beta1, gamma=gamma1, sigma=None, gamma_grad=None, beta_grad=None, gamma_optimizer=gamma1_optimizer, beta_optimizer=beta1_optimizer)
relu1 = ReLU()
maxpool1 = MaxPool()

# Convolutional layer 2
# Initializing parameters for conv2
weight2 = xavier_init(np.zeros(shape=(32, 16, 3, 3), dtype=np.float32), "Uniform")
bias2 = np.zeros(shape=(32,), dtype=np.float32)
weight2_optimizer = Adam(weight2)
bias2_optimizer = Adam(bias2)

# Initializing parameters for batchnorm2
gamma2 = np.ones(shape=(32,), dtype=np.float32)
beta2 = np.zeros(shape=(32,), dtype=np.float32)
gamma2_optimizer = Adam(gamma2)
beta2_optimizer = Adam(beta2)

conv2 = Conv2D(weight2, bias2, None, weight2_optimizer, bias2_optimizer)
batchnorm2 = BatchNorm2D(epsilon=1e-5, beta=beta2, gamma=gamma2, sigma=None, gamma_grad=None, beta_grad=None, gamma_optimizer=gamma2_optimizer, beta_optimizer=beta2_optimizer)
relu2 = ReLU()
maxpool2 = MaxPool()

# Flatten it to feed into fully connected
flatten = Flatten()

# Fully Connected layer
weight3 = xavier_init(np.zeros(shape=(1568, 10), dtype=np.float32), "Uniform")
bias3 = np.zeros(shape=(10,), dtype=np.float32)
weight3_optimizer = Adam(weight3)
bias3_optimizer = Adam(bias3)
fc = FullyConnected(weight3, bias3, weight3_optimizer, bias3_optimizer)

# gathering the data for MNIST and assigning it to variables
mnist =  fetch_openml('mnist_784', version=1, as_frame=False)

x_train = mnist.data[:2500].reshape(2500, 1, 28, 28)  # temp change to 5000 for the sake of testing
x_train = x_train / 255.0
y_train = mnist.target[:2500].astype(int)
y_train = np.eye(10)[y_train]

x_test = mnist.data[60000:].reshape(-1, 1, 28, 28)
x_test = x_test / 255.0
y_test = mnist.target[60000:].astype(int)
y_test = np.eye(10)[y_test]

# Creating the layers
layers = [conv1, batchnorm1, relu1, maxpool1, conv2, batchnorm2, relu2, maxpool2, flatten, fc]

model = Model(layers)

epochs = 3
batch_size = 32

num_batches = len(x_train) // batch_size  # 32 is the batch size

for e in range(epochs):  # looping over the epochs
    for b in range(num_batches):  # goes over the batches
        # forward prop
        predictions = model.forward(x_train[b*batch_size: (b+1)*batch_size])
        probs = softmax(predictions)

        # backward prop
        loss = cross_entropy_loss(y_train[b*batch_size: (b+1)*batch_size], probs, batch_size)
        grad = probs - y_train[b*batch_size: (b+1)*batch_size]
        model.backward(grad)
        model.update()

        predicted_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(y_train[b * batch_size: (b + 1) * batch_size], axis=1)
        accuracy = np.mean(predicted_classes == true_classes)

        print(f"epoch: {e+1},   batch: {b},   accuracy: {round(accuracy, 3)},   loss: {round(loss,3)},   grad: {round(np.max(np.abs(grad)), 3)}")



