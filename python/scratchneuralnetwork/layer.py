from .tensor import Tensor
from enum import Enum
import math
import random


class ActiFun(Enum):
    NONE = None
    SIGMOID = 1
    RELU = 2

    def _limit(self, nin: int, nout: int) -> float:
        match self:
            case ActiFun.RELU:
                return math.sqrt(2.0 / nin)
            case _:
                return math.sqrt(6.0 / (nin + nout))

    def forward(self, x: Tensor) -> Tensor:
        match self:
            case ActiFun.SIGMOID:
                return Tensor._from_impl(x._impl.sigmoid_tensor(), None, [x])
            case ActiFun.RELU:
                return Tensor._from_impl(x._impl.relu_tensor(), None, [x])
            case _:
                return x

    def derivative(self, x: Tensor) -> Tensor:
        match self:
            case ActiFun.SIGMOID:
                return Tensor._from_impl(x._impl.sigmoid_derivative_tensor(), None, [x])
            case ActiFun.RELU:
                return Tensor._from_impl(x._impl.relu_derivative_tensor(), None, [x])
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

        limit = acti_fun._limit(nin, nout)
        data = [[random.uniform(-limit, limit) for _ in range(nin)] for _ in range(nout)]
        self.weights: Tensor = Tensor(data)
        self.biases: Tensor = Tensor.zeros(nout, 1)
        self.x: Tensor = None
        self.z: Tensor = None
        self.a: Tensor = None
        self.da: Tensor = None
        self.delta_z: Tensor = None

    def forward(self, tin: Tensor) -> None:
        self.x = tin
        self.z = Tensor._from_impl(
            self.weights._impl.dot_add_bias_tensor(self.x._impl, self.biases._impl),
            None, [self.weights, self.x, self.biases],
        )
        self.a = self.acti_fun.forward(self.z)
        self.da = self.acti_fun.derivative(self.z)

    def backward(self, learning_rate: float) -> None:
        grad_w = self.delta_z @ self.x.transpose()
        grad_b = self.delta_z.sum_cols()

        self.weights = self.weights - grad_w * learning_rate
        self.biases = self.biases - grad_b * learning_rate

    def __repr__(self):
        return f"Layer(nin={self.nin}, nout={self.nout}, acti_fun={self.acti_fun})"
