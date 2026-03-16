# Scratch Neural Network
A simple neural network library written in C++ for educational purposes. It provides basic functionality to create and train neural networks, including dense layers, activation functions, and backpropagation.

## Building the project and running tests
Simply run `make`. It will build the projects and run tests.<br>
If you only want to build the project, run `make build`.

# Todo
- Add Python bindings
- Add CUDA stuff
- Store weights (Save and load models)
- Activation functions
- Up next: Mudar layers e models para Python

BRANCH "python-migrations" FAZ 2 GRANDES COISAS
1. TORNAR FORMATO DO PROJETO EM EXTENSÃO DE PYTHON
2. MUDAR LAYER & MODEL P/ PYTHON
3. opcional: melhorar CMakeLists

# Roadmap
- Improve Makefiles and Integration with VSCode Tasks (Pressing F5 gives multiple options, running Python tests, running C++ tests, building, cleaning build, etc...)
- Work on simple Python bindings
- Cleaner code
- Integrate CUDA
- Auto amount of layers and neurons: https://chatgpt.com/c/69ab0e56-dfb0-832d-bef3-b8e894c9f9da

# Structure
data/ - Load data
math/ - Matrix operations, should include CUDA support
network/ - FNN classes
python/ - Python bindings

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