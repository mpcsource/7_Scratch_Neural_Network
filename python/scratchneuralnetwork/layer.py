from .tensor import Tensor
from enum import Enum


class ActiFun(Enum):
    NONE = None
    SIGMOID = 1
    RELU = 2

    def forward(self, x: Tensor) -> Tensor:
        match self:
            case ActiFun.SIGMOID:
                return 1 / (1 + (-x).exp())
            case ActiFun.RELU:
                return x.relu()
            case _:
                return x

    def derivative(self, x: Tensor) -> Tensor:
        match self:
            case ActiFun.SIGMOID:
                s = self.forward(x)
                return s * (1 - s)
            case ActiFun.RELU:
                return x.relu_derivative()
            case _:
                return Tensor.ones(*x.shape)


class Layer:
    """Layer class."""

    def __init__(
        self,
        nin: int = 0,
        nout: int = 0,
        acti_fun: ActiFun = ActiFun.NONE,
    ) -> None:

        self.nin: int = nin
        self.nout: int = nout
        self.acti_fun: ActiFun = acti_fun
        self.weights: Tensor = Tensor.zeros(nout, nin)
        self.biases: Tensor = Tensor.zeros(nout, 1)
        self.x: Tensor = None
        self.z: Tensor = None
        self.a: Tensor = None
        self.da: Tensor = None

    def forward(self, tin: Tensor) -> None:
        self.x = tin
        self.z = self.weights @ self.x + self.biases
        self.a = self.acti_fun.forward(self.z)
        self.da = self.acti_fun.derivative(self.z)

    def __repr__(self):
        return f"Layer(nin={self.nin}, nout={self.nout}, acti_fun={self.acti_fun})"
