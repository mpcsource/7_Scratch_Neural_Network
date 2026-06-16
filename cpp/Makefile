.PHONY: help build clean rebuild test test-python test-cpp rebuild-test-python rebuild-test-cpp

PYTHON_EXE = $(shell grep -s '^PYTHON_EXECUTABLE' build/CMakeCache.txt 2>/dev/null | head -1 | cut -d= -f2 || echo python3)

help:
	@echo "Available targets:"
	@echo "  make build                    - Build the project"
	@echo "  make clean                    - Remove build artifacts"
	@echo "  make rebuild                  - Clean and rebuild"
	@echo "  make test-python              - Run Python tests"
	@echo "  make test-cpp                 - Run C++ tests"
	@echo "  make test                     - Run all tests (Python + C++)"
	@echo "  make rebuild-test-python      - Rebuild and run Python tests"
	@echo "  make rebuild-test-cpp         - Rebuild and run C++ tests"

build:
	clear
	cmake -S . -B build
	cmake --build build -- -j$(nproc)
	rm -f ../python/scratchneuralnetwork/_core.*.so
	cp build/_core.cpython-*.so ../python/scratchneuralnetwork/

clean:
	rm -rf build

rebuild: clean build

test-python:
	find ../python ../tests -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
	PYTHONPATH=../python $(PYTHON_EXE) ../tests/python_tests.py

test-cpp:
	ctest --test-dir build -V

test: test-python test-cpp

rebuild-test-python: rebuild test-python

rebuild-test-cpp: rebuild test-cpp