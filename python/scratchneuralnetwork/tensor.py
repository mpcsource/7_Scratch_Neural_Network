from __future__ import annotations
from enum import Enum
from typing import Union

from ._core import CTensor


class Operation(Enum):
    ADDITION = 1
    SUBTRACTION = 2
    MULTIPLICATION = 3
    MULTIPLICATION_NUMBER = 4
    DOT = 5


class Tensor:
    """Python Tensor: holds a CTensor backend + autograd metadata."""

    def __init__(self, rows: int = 0, cols: int = 0, requires_grad: bool = False):
        self._impl: CTensor = CTensor(rows, cols)
        self.op: Operation | None = None
        self.parents: list[Tensor] | None = None
        self.requires_grad: bool = requires_grad

    @staticmethod
    def _from_impl(impl: CTensor, op: Operation, parents: list[Tensor]) -> Tensor:
        """Wrap a C++ CTensor result into a Tensor with zero extra copies."""
        t = Tensor.__new__(Tensor)  # bypasses __init__, no CTensor allocation
        t._impl = impl
        t.op = op
        t.parents = parents
        t.requires_grad = False
        return t

    @property
    def grad(self) -> Tensor:
        """Return the gradient buffer (owned by C++) as a Tensor view."""
        return Tensor._from_impl(self._impl.get_grad(), None, None)

    def zero_grad(self) -> None:
        """Zero out the C++ gradient buffer."""
        self._impl.zero_grad()

    def accumulate_grad(self, incoming: Tensor) -> None:
        """Accumulate gradient: grad += incoming (delegated to C++)."""
        self._impl.accumulate_grad(incoming._impl)

    def __repr__(self) -> str:
        return f"Tensor()"

    # ===============
    # Math operations
    # ===============

    # Addition
    def __add__(self, other: Tensor) -> Tensor:

        return Tensor._from_impl(
            self._impl.add_tensor(other._impl),
            Operation.ADDITION,
            [self, other],
        )

    # Subtraction
    def __sub__(self, other: Tensor) -> Tensor:
        return Tensor._from_impl(
            self._impl.sub_tensor(other._impl),
            Operation.SUBTRACTION,
            [self, other],
        )

    # Element-wise multiplication (Tensor) and scalar multiplication (number)
    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor._from_impl(
                self._impl.mul_tensor(other._impl),
                Operation.MULTIPLICATION,
                [self, other],
            )
        if isinstance(other, (float, int)):
            return Tensor._from_impl(
                self._impl.mul_tensor_number(float(other)),
                Operation.MULTIPLICATION_NUMBER,
                [self],
            )
        raise TypeError(f"Unsupported operand type for *: {type(other)}")

    # Dot product
    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor._from_impl(
            self._impl.dot_tensor(other._impl),
            Operation.DOT,
            [self, other],
        )
