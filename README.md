# Scratch Neural Network
A simple neural network library written in C++ for educational purposes. It provides basic functionality to create and train neural networks, including dense layers, activation functions, and backpropagation.

## Building the project and running tests
Simply run `make`. It will build the projects and run tests.<br>
If you only want to build the project, run `make build`.

## Version log

### 0.1 (Current)
* Functioning layers and dense model.
* Mean squared error loss function.
* Sigmoid and ReLU activation functions.
* Glorot/He initialization.
* Forward pass.
* Batch gradient descent.
* Loads and splits data.
* Data normalization.
* Basic matrix operations.
* Pseudorandom number generator.
* Basic testing.
Provides a simple API for creating dense layers and neural networks, with backpropagation and forward pass functionality.
Recommended to check the `tests` directory for examples of how to use the API.


### 0.2 (Planned)
* Softmax activation function.
* Cross-entropy loss function.
* Mini-batch and stochastic gradient descent.
* Extensive testing.
* A more complete research paper.
* Additional checks and assertions.

#### 0.3 (Future)
* Convolutional layers.
* Adam optimizer.
* L1 and L2 regularization.
* Dropout layer.