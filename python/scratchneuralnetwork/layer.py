from .tensor import Tensor


class Layer:
    """Layer class."""
    

    # Single constructor
    def __init__(
        self, 
        nin:int = 0,
        nout:int = 0,    
    ):
        self.nin = nin
        self.nout = nout
        self.weights = Tensor(nout, nin)
        self.biases = Tensor(nout, 1)
        self.x = None # Unmodified input
        self.z = None # Output before actino function
        self.a = None # Output after activation function
        self.da = None # Derivative of activation function

    # ==========
    # Operations
    # ==========

    def forward(
        self,
        tin:Tensor, # Tensor input
    ) -> None:
        self.x = tin
        self.z = self.weights @ self.x + self.biases
        self.a = self.z
        self.da = Tensor(0, 0)

    def __repr__(self):
        return f"Layer()"
