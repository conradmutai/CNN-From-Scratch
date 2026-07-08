import numpy as np

from sklearn.datasets import fetch_openml

from .activations import softmax
from .layers import Conv2D, ReLU, MaxPool, Flatten, FullyConnected, BatchNorm2D
from .losses import cross_entropy_loss
from .optimizers import Adam
from .initializers import xavier_init
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
batchnorm1 = BatchNorm2D(epsilon=1e-5, beta=beta1, gamma=gamma1, gamma_optimizer=gamma1_optimizer, beta_optimizer=beta1_optimizer)
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
batchnorm2 = BatchNorm2D(epsilon=1e-5, beta=beta2, gamma=gamma2, gamma_optimizer=gamma2_optimizer, beta_optimizer=beta2_optimizer)
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
mnist = fetch_openml('mnist_784', version=1, as_frame=False)


def train(x_train, y_train, epochs):
    # Creating the layers
    layers = [conv1, batchnorm1, relu1, maxpool1, conv2, batchnorm2, relu2, maxpool2, flatten, fc]

    model = Model(layers)

    # make sure BatchNorm is in training mode in case test() was called before this
    for layer in model.layers:
        if isinstance(layer, BatchNorm2D):
            layer.training = True

    batch_size = 32

    num_batches = len(x_train) // batch_size  # 32 is the batch size

    loss_history = []  # a list containing the losses per each epoch
    train_acc_history = []  # a list containing the accuracy per each epoch
    predicted_classes_history = []
    true_classes_history = []

    for e in range(epochs):  # looping over the epochs
        # shuffle the training data each epoch
        perm = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_losses = []
        epoch_accs = []

        for b in range(num_batches):  # goes over the batches
            batch_x = x_train_shuffled[b*batch_size: (b+1)*batch_size]
            batch_y = y_train_shuffled[b*batch_size: (b+1)*batch_size]

            # forward prop
            predictions = model.forward(batch_x)
            probs = softmax(predictions)

            # backward prop
            loss = cross_entropy_loss(batch_y, probs, batch_size)
            grad = probs - batch_y
            model.backward(grad)
            model.update()

            predicted_classes = np.argmax(probs, axis=1)
            true_classes = np.argmax(batch_y, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)

            epoch_losses.append(loss)
            epoch_accs.append(accuracy)

            # only keep predictions from the final epoch, to avoid unbounded memory growth
            if e == epochs - 1:
                predicted_classes_history.append(predicted_classes)
                true_classes_history.append(true_classes)

        loss_history.append(np.mean(epoch_losses))
        train_acc_history.append(np.mean(epoch_accs))
        print(f"epoch: {e + 1}   loss: {round(np.mean(epoch_losses), 3)}   train acc: {round(np.mean(epoch_accs), 3)}")

    return loss_history, train_acc_history, predicted_classes_history, true_classes_history


def test(x_test, y_test):
    layers = [conv1, batchnorm1, relu1, maxpool1, conv2, batchnorm2, relu2, maxpool2, flatten, fc]
    model = Model(layers)

    batch_size = 32

    # switching the batch norm into test phase to allow proper passes through the CNN
    for layer in model.layers:
        if isinstance(layer, BatchNorm2D):
            layer.training = False

    try:
        total_correct = 0
        total_seen = 0

        num_test_batch = len(x_test) // batch_size

        for b in range(num_test_batch):
            test_batch_image = x_test[b * batch_size: (b + 1) * batch_size]
            test_batch_labels = y_test[b * batch_size: (b + 1) * batch_size]

            predictions = model.forward(test_batch_image)
            probs = softmax(predictions)
            predicted_classes = np.argmax(probs, axis=1)
            true_classes = np.argmax(test_batch_labels, axis=1)
            total_correct += np.sum(predicted_classes == true_classes)
            total_seen += batch_size

        test_accuracy = total_correct / total_seen
        test_accuracy_percentage = round(test_accuracy * 100, 3)
        print(f"test accuracy: {test_accuracy_percentage}%")

        return test_accuracy
    finally:
        # always restore training mode, even if something above raises
        for layer in model.layers:
            if isinstance(layer, BatchNorm2D):
                layer.training = True


def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


