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