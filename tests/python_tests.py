import scratchneuralnetwork as snn
import math
import faulthandler
faulthandler.enable()

def mae(true, pred, n):
    total = 0
    for i in range(n):
        total += abs(true[i, 0] - pred[i, 0])
    return total / n

def rmse(true, pred, n):
    total = 0
    for i in range(n):
        diff = true[i, 0] - pred[i, 0]
        total += diff * diff
    return math.sqrt(total / n)

def r2(true, pred, n):
    mean = sum(true[i, 0] for i in range(n)) / n
    ss_tot = sum((true[i, 0] - mean) ** 2 for i in range(n))
    ss_res = sum((true[i, 0] - pred[i, 0]) ** 2 for i in range(n))
    return 1 - (ss_res / ss_tot)

# Load data from CSV file.
print("Loading data...")
data = snn.load_data("tests/data.csv", ",", True)
print(f"Loaded: {data.rows} rows, {data.cols} cols")

# Split data into training and testing sets.
print("Splitting data...")
x_train, y_train, x_test, y_test = snn.train_test_split(data, 8)
print(f"x_train: {x_train.rows}x{x_train.cols}, x_test: {x_test.rows}x{x_test.cols}")

# Normalize the data.
print("Normalizing...")
x_train, mean_x, std_x = snn.normalize_data(x_train)
x_test = snn.normalize_data(x_test, mean_x, std_x)

# Normalize the target variable.
y_train, mean_y, std_y = snn.normalize_data(y_train)
y_test = snn.normalize_data(y_test, mean_y, std_y)
print("Done normalizing.")

# Create 4 layers.
l1 = snn.Layer(9, 256, "relu")
l2 = snn.Layer(256, 128, "relu")
l3 = snn.Layer(128, 64, "relu")
l4 = snn.Layer(64, 1, "linear")

# Create model.
model = snn.Model("mse", snn.Decay.COSINE, lr_min=1e-5)
model.append_layer(l1)
model.append_layer(l2)
model.append_layer(l3)
model.append_layer(l4)

# Backpropagate the model.
print("Training...")
model.backprop(x_train, y_train, 300, 0.01, 256)
print("Training done.")

# Unnormalize.
print("Unnormalizing...")
y_test = snn.unnormalize_data(y_test, mean_y, std_y)
y_train = snn.unnormalize_data(y_train, mean_y, std_y)
print("Done.")

# Print results.
print("True:")
y_test.tail().print()

print("Pred:")
y_hat = model.test(x_test, y_test)
y_hat = snn.unnormalize_data(y_hat, mean_y, std_y)
y_hat.tail().print()
print("All done.")

# Metrics
n = y_test.rows
print(f"\nMAE:  {mae(y_test, y_hat, n):.2f}")
print(f"RMSE: {rmse(y_test, y_hat, n):.2f}")
print(f"R²:   {r2(y_test, y_hat, n):.4f}")