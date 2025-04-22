# 7_Scratch_Neural_Network
 
1. How to build the project:
From ./:
`cmake -S . -B build`

2. How to run the project:
`cmake --build build`

3. How to run tests
`ctest --test-dir build -V`

* All at the same time:
`clear && cmake -S . -B build && cmake --build build && ctest --test-dir build -V`

# Layer vs Model
* Layers process one input at a time, models handle multiple, and they facilitate passing data from one layer to another.

# Versions

## Alpha 0.1 (Current)
* A functioning dense-layered sequential linear neural network that can learn (backpropagation) and predict (forwardpass).
* Loads data, does matrix operations, has a pseudorandom number generator.
* Has simple testing.

## Alpha 0.2 (Planned)
* Much better testing.
* Extensive testing.
* The academic/research paper.
* All implemented, integrated together:
    * Activation functions.
    * Glorot initialization.
    * Error/loss/cost functions.
* Asserts in every method.

## Future
* Visual recognition?