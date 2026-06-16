import scratchneuralnetwork as snn

# Dataset
dataset = snn.Dataset(
    path="/home/miguel/7_Scratch_Neural_Network/tests/data.csv",
    target_column="median_house_value",
)
print("Features:", dataset.processed_feature_names)
print("X shape:", dataset.X.shape)
print("y shape:", dataset.y.shape)
print("Encodings:", dataset.encodings)


n_features = 13
model = snn.Model(layers=[
    snn.Layer(n_features, 64, snn.ActiFun.NONE),
    snn.Layer(64, 64, snn.ActiFun.NONE),
    snn.Layer(64, 1, snn.ActiFun.NONE),
])

model.backprop(dataset.X, dataset.y, epochs=3, learning_rate=0.01)

predictions = model.forward(dataset.X)
print("Predictions stats:")
print("  min:", min(predictions.flat))
print("  max:", max(predictions.flat))

predictions_original = dataset.unnormalize_y(predictions)
print("Unnormalized predictions stats:")
print("  min:", min(predictions_original.flat))
print("  max:", max(predictions_original.flat))
