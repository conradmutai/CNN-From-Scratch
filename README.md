# Convolutional Neural Network from Scratch
Creating a CNN without the usage of core tensor libraries such as **PyTorch** or **TensorFlow**, while mainly utilizing **NumPy**. Implements forward pass, backpropagation, and gradient descent following essential principles.

## What This Is
A Convolutional Neural Network made with NumPy without the use of any external machine learning libraries. This project was created as a display and test of my knowledge of rudimentary deep learning concepts, ranging from the basics, such as weight initialization, to the complexities of convolutional neural networks and backpropagation to achieve an optimal model loss. In addition, it will be benchmarked against a model built with PyTorch on the MNIST dataset, a dataset commonly used to test and benchmark model efficiency, which in turn can inform further development of the model.

## Architecture
[ Conv2D -> ReLU -> MaxPool] * 2 -> Flatten -> FC -> Softmax

## Results
### Metric
Test Accuracy and Train Accuracy in %, Loss in decimal.

### This Model
Test Accuracy: 98%
Train Accuracy (Highest): 100%
Loss: 0.0

## Design Decisions
For the MNIST model, I chose to utilize:
1 -> 32 -> 64: as MNIST images are quite simple, as they maintain themselves in a gray-scale low-detail format, the reason for doubling 32 to 64 is that it allows the second convolutional layer to combine the edge detectors from the first layer.
64 -> 128: This is done after pooling, as once spatial resolution drops, increasing channels tends to compensate by capturing more patterns per spatial location, following the pattern used in CNN designs such as ResNet, where halving the resolution doubles the channels.

## How to Run
To run this model, a demo notebook is attached and is opened through Jupyter Notebook. There's an option to "Run" at the top right, and selecting "Run All" will allow the model to run properly without any issues. As a warning, this model takes quite a while to train based on the system that's being used, as it isn't GPU-accelerated like libraries such as PyTorch.

## What I learned
I maintained a solid understanding of a Convolutional Neural Network, and the operation of kernel sizes that apply weights to selected hidden units of a previous layer, as well as what it means for a complete forward pass, but where I feel that I always lacked was the proper understanding of backpropagation, and specifically that for a CNN. It required me to go back and read how the operation of backprop works with this, and led to me having to differentiate different equations to get it working. It took trial and error, taking several days to resolve, and along the way I ran into several real bugs that forced me to actually understand what was happening rather than just accepting that the code worked. 

One issue was a dead ReLU problem, where too many units stopped activating and just carried zero gradient forward. Another was my BatchNorm layer not properly storing running stats, which meant training looked fine but inference completely broke down. 

I also found that I had hardcoded stride=1 into my Conv2D backward pass without realizing it, which meant gradients were wrong for any layer using a different stride. The hardest one to catch was in my Adam optimizer, where I had written the bias correction as 1/(beta**t) instead of the correct formulation, which quietly skewed the whole training process without throwing any errors.

At the end, I was able to come to a proper conclusion, having gone through backprop for a CNN by hand and fixed each of these issues individually.

## References

1. GeeksforGeeks. *Backpropagation in Convolutional Neural Networks*. https://www.geeksforgeeks.org/computer-vision/backpropagation-in-convolutional-neural-networks/
2. GeeksforGeeks. *Xavier Initialization*. https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
3. GeeksforGeeks. *One Hot Encoding*. https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
4. APXML. *Kaiming (He) Initialization – How to Build a Large Language Model, Ch. 12*. https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization
