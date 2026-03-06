.PHONY: help build clean rebuild test test-python test-cpp rebuild-test-python rebuild-test-cpp

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

clean:
	rm -rf build

rebuild: clean build

test-python:
	PYTHONPATH=build python3 tests/python_tests.py

test-cpp:
	ctest --test-dir build -V

test: test-python test-cpp

rebuild-test-python: rebuild test-python

rebuild-test-cpp: rebuild test-cpp