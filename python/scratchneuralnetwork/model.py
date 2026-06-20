import random
from .tensor import Tensor
from .layer import Layer


class Model:
    """Model class."""

    def __init__(
            self,
            layers: list[Layer] = None,
            ) -> None:

        self.layers: list[Layer] = layers if layers is not None else []

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def forward(
            self,
            x: Tensor,
            ) -> Tensor:

        for layer in self.layers:
            layer.forward(x)
            x = layer.a
        return x

    def backward(
            self,
            label: Tensor,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            ) -> None:

        # Output layer delta: d(MSE)/da * da * (1/batch_size)
        out_layer = self.layers[-1]
        dloss = out_layer.a - label
        out_layer.delta_z = dloss * out_layer.da * (1.0 / batch_size)

        # Propagate delta through hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            next_layer = self.layers[i + 1]
            layer = self.layers[i]
            layer.delta_z = (next_layer.weights.transpose() @ next_layer.delta_z) * layer.da

        # Update weights and biases
        for layer in self.layers:
            layer.backward(learning_rate)

    def backprop(
            self,
            data: Tensor,
            labels: Tensor,
            epochs: int = 10,
            learning_rate: float = 0.01,
            batch_size: int = None,
            shuffle: bool = True,
            ) -> None:

        # data shape is (n_features, n_samples)
        n_samples = data.shape[1] if len(data.shape) > 1 else 1
        if batch_size is None:
            batch_size = n_samples

        steps = n_samples // batch_size

        for epoch in range(epochs):
            epoch_data = data
            epoch_labels = labels

            if shuffle:
                indices = list(range(n_samples))
                random.shuffle(indices)
                epoch_data = data.gather_cols(indices)
                epoch_labels = labels.gather_cols(indices)

            for step in range(steps):
                start = step * batch_size
                batch_x = epoch_data.slice_cols(start, batch_size)
                batch_y = epoch_labels.slice_cols(start, batch_size)
                self.forward(batch_x)
                self.backward(batch_y, learning_rate, batch_size)

    def append_layer(
            self,
            layer: Layer = None,
            ) -> None:

        if layer is not None and type(layer) is Layer and layer not in self.layers:
            self.layers.append(layer)