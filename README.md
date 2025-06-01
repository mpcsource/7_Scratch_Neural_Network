# Scratch Neural Network
A simple neural network library written in C++ for educational purposes. It provides basic functionality to create and train neural networks, including dense layers, activation functions, and backpropagation.

(The name really should be changed)

## Prerequisites
* C++20 compatible compiler
* CMake 3.14 or newer
* GNU Make

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mpcsource/7_Scratch_Neural_Network.git
   cd 7_Scratch_Neural_Network
   ```


## Building the project and running tests
Simply run `make`. It will build the projects and run tests.<br>
If you only want to build the project, run `make build`.

## Example usage

### Loading data
```cpp
#include "snn.h"
int main() {
   Matrix data = loadData("data.csv", ",", "true");
   data.head().basicPrint();
}
```

Check the `tests` directory for more examples of how to use the API.

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
Note: I am actually stupid and named this branch `v0.3` instead of `v0.2`, so the next version will be `v0.3` instead of `v0.2`.
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