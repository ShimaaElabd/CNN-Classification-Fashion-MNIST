# CNN for Classification Fashion MNIST

![download](https://user-images.githubusercontent.com/25883512/52514138-0e2a4800-2c18-11e9-80c1-ae93a2dcd51f.jpg)

# CNN for Classification Fashion-MNIST:
---

* In this notebook, I define and train an CNN to classify images from the [Fashion-MNIST database](https://github.com/zalandoresearch/fashion-mnist). 

* This CNN with two convolutional layers and additional fully-connected and dropout layers to avoid overfitting the data and gradient descent with momentum to avoid reaching a local minimum.

### Define the network architecture

The various layers that make up any neural network are documented, [here](http://pytorch.org/docs/master/nn.html). For a convolutional neural network, will use a simple series of layers:
* Convolutional layers
* Maxpooling layers
* Fully-connected (linear) layers

Adding [dropout layers](http://pytorch.org/docs/stable/nn.html#dropout) to avoid overfitting this data.

---

To define a neural network in PyTorch, we define the layers of a model in the function `__init__` and define the feedforward behavior of a network that employs those initialized layers in the function `forward`, which takes in an input image tensor, `x`. The structure of this Net class is shown below.

During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.

#### Define the Layers in ` __init__`
a conv/pool layer may be defined like this (in `__init__`):
```
# 1 input image channel (for grayscale images), 32 output channels/feature maps, 3x3 square convolution kernel
self.conv1 = nn.Conv2d(1, 32, 3)

# maxpool that uses a square window of kernel_size=2, stride=2
self.pool = nn.MaxPool2d(2, 2)      
```

#### Refer to Layers in `forward`
Then referred to in the `forward` function like this, in which the conv1 layer has a ReLu activation applied to it before maxpooling is applied:
```
x = self.pool(F.relu(self.conv1(x)))
```

Place any layers with trainable weights, such as convolutional layers, in the `__init__` function and refer to them in the `forward` function; any layers or functions that always behave in the same way, such as a pre-defined activation function, may appear *only* in the `forward` function. In practice, often see conv/pool layers defined in `__init__` and activations defined in `forward`.

#### Convolutional layer
It takes in a 1 channel (grayscale) image and outputs 10 feature maps as output, after convolving the image with 3x3 filters.

#### Flattening

To move from the output of a convolutional/pooling layer to a linear layer,  first flatten the extracted features into a vector. In Keras, this done by `Flatten()`, and in PyTorch, flatten an input `x` with `x = x.view(x.size(0), -1)`.

### Define the rest of the layers

* Use at least two convolutional layers
* Output must be a linear layer with 10 outputs (for the 10 classes of clothing)
* Use a dropout layer to avoid overfitting

### A note on output size

For any convolutional layer, the output feature maps will have the specified depth (a depth of 10 for 10 filters in a convolutional layer) and the dimensions of the produced feature maps (width/height) can be computed as the _input image_ width/height, W, minus the filter size, F, divided by the stride, S, all + 1. The equation looks like: `output_dim = (W-F)/S + 1`, for an assumed padding size of 0. Derivation of this formula, [here](http://cs231n.github.io/convolutional-networks/#conv).

For a pool layer with a size 2 and stride 2, the output dimension will be reduced by a factor of 2. 
