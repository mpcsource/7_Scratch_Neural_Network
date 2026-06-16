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
            ) -> None:

        if batch_size is None:
            batch_size = data.shape[0]

        # TODO: implement mini-batch batching when Tensor supports slicing.
        # For now, runs full-batches only (batch_size = n_samples).
        n_samples = data.shape[0]
        steps = n_samples // batch_size

        for epoch in range(epochs):
            for step in range(steps):
                # TODO: extract proper batch slices
                label_hat = self.forward(data)
                self.backward(labels, learning_rate, batch_size)

    def append_layer(
            self,
            layer: Layer = None,
            ) -> None:

        if layer is not None and type(layer) is Layer and layer not in self.layers:
            self.layers.append(layer)