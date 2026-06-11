import scratchneuralnetwork as snn

# Dataset
dataset = snn.Dataset(path="/home/miguel/7_Scratch_Neural_Network/tests/data.csv")
print(dataset)
print(dataset.head())



m = snn.Model(layers=[
    snn.Layer(),
])

print("Model:", m)
