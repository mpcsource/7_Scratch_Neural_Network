from .tensor import Tensor
from .layer import Layer

class Model:
    """Model class."""
    
    def __init__(
        self,
    ):
        self.layers = []

    def __repr__(self):
        return f"Model()"

    def append_layer(
        self,
        layer: Layer = None,
    ):
        if layer is not None:
            self.layers.append(layer)