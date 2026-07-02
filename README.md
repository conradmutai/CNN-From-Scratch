# Convolutional Neural Network from Scratch
Creating a CNN without the usage of core tensor libraries such as **PyTorch** or **TensorFlow**, while mainly utilizing **NumPy**. Implements forward pass, backpropagration, and gradient descent following essential principles.

## What This Is
A Convolutional Neural Network made with NumPy without the usage of any external machine learning libraries. This project was created as a display and test of my knowledge of rudimentary deep learning concepts, ranging from the basics, such as weight initialization, to the complexities of convolutional neural networks and backpropagation to achieve an optimal model loss. In addition, it will be benchmarked against a model made using PyTorch, with the MNIST data set, which is commonly used in testing/benchmarking a model's efficiency, which in turn can lead to further development of the model.

## Architecture
[ Conv2D -> ReLU -> MaxPool] * 3 -> Flatten -> FC -> Softmax

## Results
### Metric

### This Model

## Design Decisions


## How to Run
To run this model, a demo notebook is attached and is opened through Jupyter Notebook. There's an option to "Run" at the top right, and selecting "Run All" will allow the model to run properly without any issues. As a warning, this model takes quite a while to train based on the system that's being used, as it isn't GPU-accelerated like libraries such as PyTorch.

## What I learned


## References

1. GeeksforGeeks. *Backpropagation in Convolutional Neural Networks*. https://www.geeksforgeeks.org/computer-vision/backpropagation-in-convolutional-neural-networks/
2. GeeksforGeeks. *Xavier Initialization*. https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
3. GeeksforGeeks. *One Hot Encoding*. https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
4. APXML. *Kaiming (He) Initialization – How to Build a Large Language Model, Ch. 12*. https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization
