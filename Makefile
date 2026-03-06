.PHONY: all python build test clean

# Builds and tests the project
all: build test

# Build the project using CMake
build:
	clear
	cmake -S . -B build
	cmake --build build

# Run tests using CTest
test:
	ctest --test-dir build -V

# Run Python tests
python:
	python3 tests/python_tests.py

# Clean the build directory
clean:
	rm -rf build