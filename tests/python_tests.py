import scratchneuralnetwork as snn

matrix1 = snn.Matrix(10, 10, 1.0)
matrix2 = snn.Matrix(10, 10, 1.0)
matrix3 = matrix1 @ matrix2

matrix3.print()