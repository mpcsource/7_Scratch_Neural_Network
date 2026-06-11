from .tensor import Tensor
from .layer import Layer

class Model:
    """Model class."""
    
    def __init__(
            self,
            layers: list[Layer] = [],
            ) -> None:
        
        self.layers: list[Layer] = layers

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def forward(
            self,
            x: Tensor = None,
            ) -> None:
        
        for layer in self.layers:
            layer.forward(x)    

    def append_layer(
            self,
            layer: Layer = None,
            ) -> None:
        
        if layer is not None and type(layer) is Layer and layer not in self.layers:
            self.layers.append(layer)