from __future__ import annotations
from enum import Enum
from typing import Union

from ._core import CTensor

class Operations(Enum):
    ADDITION = 1
    SUBTRACTION = 2
    MULTIPLICATION = 3
    MULTIPLICATION_NUMBER = 4
    DOT = 5

class Tensor(CTensor):
    """Python wrapper around the C++ Tensor backend."""

    # Single constructor
    def __init__(self, rows: Union[int, CTensor] = 0, cols:int = 0):
        if isinstance(rows, CTensor):
            super().__init__(rows)
        else:
            super().__init__(rows, cols)
        self.op = None # Operation that generated this tensor
        self.parents = None # Elements part of the operation that generated this tensor

    @staticmethod
    def _from_cpp(cpp_tensor: CTensor, op: Operations, parents: list[Tensor]) -> Tensor:
        out = Tensor(cpp_tensor)
        out.op = op
        out.parents = parents
        return out
    
    def __repr__(self):
        return f"Tensor()"
    
    # ===============
    # Math operations
    # ===============

    # Addition
    def __add__(self, other: Tensor) -> Tensor:
        out_cpp = self.add_tensor(other)
        return Tensor._from_cpp(out_cpp, Operations.ADDITION, [self, other])
    
    # Subtraction
    def __sub__(self, other: Tensor) -> Tensor:
        out_cpp = self.sub_tensor(other)
        return Tensor._from_cpp(out_cpp, Operations.SUBTRACTION, [self, other])
    
    # Element-wise multiplication (Tensor) and scalar multiplication (number)
    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        if isinstance(other, Tensor):
            out_cpp = self.mul_tensor(other)
            return Tensor._from_cpp(out_cpp, Operations.MULTIPLICATION, [self, other])

        if isinstance(other, (float, int)):
            out_cpp = self.mul_tensor_number(float(other))
            return Tensor._from_cpp(out_cpp, Operations.MULTIPLICATION_NUMBER, [self])

        raise TypeError(f"Unsupported operand type for *: {type(other)}")
    
    # Dot product
    def __matmul__(self, other: Tensor) -> Tensor:
        out_cpp = self.dot_tensor(other)
        return Tensor._from_cpp(out_cpp, Operations.DOT, [self, other])
