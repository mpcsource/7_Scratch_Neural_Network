import scratchneuralnetwork as snn

ct = snn.CTensor()
print("CTensor:", ct)

t = snn.Tensor()
print("Tensor:", t)
print("Tensor is CTensor:", isinstance(t, snn.CTensor))

a = t + snn.Tensor()

l = snn.Layer()
print("Layer:", l)

m = snn.Model()
print("Model:", m)
