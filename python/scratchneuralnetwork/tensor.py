from __future__ import annotations
from collections.abc import Sequence
from enum import Enum
from typing import Union, Optional
import csv as _csv

from ._core import CTensor


class Operation(Enum):
    ADDITION = 1
    SUBTRACTION = 2
    MULTIPLICATION = 3
    MULTIPLICATION_NUMBER = 4
    DOT = 5


class Tensor:
    """Python Tensor: holds a CTensor backend + autograd metadata."""

    def __init__(self, data=None, requires_grad: bool = False):
        self.shape, flat = self._to_shape_and_flat(data)
        self._flat = flat
        self._impl: CTensor = CTensor(self.shape, self._flat)
        self.op: Operation | None = None
        self.parents: list[Tensor] | None = None
        self.requires_grad: bool = requires_grad

        # Dataset variables
        self.encoding: dict[str, int] | None = None

    @property
    def flat(self) -> list[float]:
        if self._flat is None:
            self._flat = list(self._impl.flat)
        return self._flat

    @staticmethod
    def _to_shape_and_flat(data) -> tuple[list[int], list[float]]:
        if data is None:
            return [], []

        if isinstance(data, (int, float)):
            return [1], [float(data)]

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError("Tensor data must be a scalar or nested sequence of numbers")

        shape: list[int] = []
        flat: list[float] = []

        def walk(node, depth: int) -> None:
            if isinstance(node, (int, float)):
                flat.append(float(node))
                return

            if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
                raise TypeError("Tensor data must contain only numbers or nested sequences")

            length = len(node)
            if depth == len(shape):
                shape.append(length)
            elif shape[depth] != length:
                raise ValueError("Ragged tensor data is not supported")

            for item in node:
                walk(item, depth + 1)

        walk(data, 0)
        return shape, flat

    @staticmethod
    def zeros(*shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if any(dim < 0 for dim in shape):
            raise ValueError("Tensor dimensions must be non-negative")

        def build(level: int):
            if level == len(shape):
                return 0.0
            return [build(level + 1) for _ in range(shape[level])]

        return Tensor(build(0))

    @staticmethod
    def _from_impl(
                impl: CTensor, 
                op: Operation, 
                parents: list[Tensor],
                encoding: dict[int, dict[str, int]] | None = None,
                ) -> Tensor:
        
        """Wrap a C++ CTensor result into a Tensor with zero extra copies."""
        t = Tensor.__new__(Tensor)  # bypasses __init__, no CTensor allocation
        t._impl = impl
        t.shape = list(impl.shape)
        t._flat = None
        t.op = op
        t.parents = parents
        t.requires_grad = False
        t.encoding = encoding
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

    def head(self, n: int = 5) -> Tensor:
        return Tensor(self.flat[:n * self.shape[1]]) if len(self.shape) >= 2 else Tensor(self.flat[:n])

    def tail(self, n: int = 5) -> Tensor:
        total = (self.shape[0] * self.shape[1]) if len(self.shape) >= 2 else self.shape[0]
        row_size = self.shape[1] if len(self.shape) >= 2 else 1
        return Tensor(self.flat[total - n * row_size:]) if len(self.shape) >= 2 else Tensor(self.flat[-n:])

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"

    @classmethod
    def from_csv(
        cls, 
        path: str,
        delimiter: str = ",",
        has_header: bool = True,
        columns: Optional[Sequence[int]] = None,
        skip_rows: int = 0,
        requires_grad: bool = False,
        ) -> Tensor:
        """
        Create a 2-D Tensor from a CSV file.

        Args:
            path: Path to CSV file.
            delimiter: Field separator.
            has_header: If true, skips first row.
            columns: Optional list of column to keep (in order). None = all.
            skip_rows: Number of leading rows to skip.

        Returns:
            A Tensor of shape [n_rows, n_cols].
        """

        rows: list[list[float]] = []

        with open(path, newline="") as f:
            reader = _csv.reader(f, delimiter=delimiter)

            for _ in range(skip_rows):
                next(reader, None)

            if has_header:
                next(reader, None)

            for lineno, raw in enumerate(reader, start=1):
                if not raw: # Skip blank lines
                    continue
                cells = raw if columns is None else [raw[i] for i in columns]

                try:
                    rows.append([float(c) for c in cells])
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Could not parse row {lineno} of {path!r}: {raw}"
                    ) from e
                
        if not rows:
            raise ValueError(f"No data rows found in {path!r}")

        return cls(rows, requires_grad=requires_grad)



    # ===============
    # Math operations
    # ===============

    # Addition
    def __add__(
            self, 
            other: Tensor
            ) -> Tensor:

        return Tensor._from_impl(
            self._impl.add_tensor(other._impl),
            Operation.ADDITION,
            [self, other],
        )

    def add_bias(self, bias: Tensor) -> Tensor:
        return Tensor._from_impl(
            self._impl.add_bias(bias._impl),
            Operation.ADDITION,
            [self, bias],
        )

    # Subtraction
    def __sub__(
            self, 
            other: Tensor
            ) -> Tensor:

        return Tensor._from_impl(
            self._impl.sub_tensor(other._impl),
            Operation.SUBTRACTION,
            [self, other],
        )

    # Element-wise multiplication (Tensor) and scalar multiplication (number)
    def __mul__(
            self, 
            other: Union[Tensor, float, int]
            ) -> Tensor:

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

    @staticmethod
    def ones(*shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if any(dim < 0 for dim in shape):
            raise ValueError("Tensor dimensions must be non-negative")

        def build(level: int):
            if level == len(shape):
                return 1.0
            return [build(level + 1) for _ in range(shape[level])]

        return Tensor(build(0))

    def transpose(self) -> Tensor:
        return Tensor._from_impl(
            self._impl.transpose_tensor(),
            None,
            [self],
        )

    def sum_cols(self) -> Tensor:
        return Tensor._from_impl(
            self._impl.sum_cols_tensor(),
            None,
            [self],
        )

    def slice_cols(self, start: int, count: int) -> Tensor:
        flat = self.flat
        nf, ns = self.shape
        if start + count > ns:
            raise ValueError(f"slice_cols: start={start}, count={count} exceeds cols={ns}")
        result = [0.0] * (nf * count)
        for i in range(nf):
            src = i * ns + start
            dst = i * count
            for j in range(count):
                result[dst + j] = flat[src + j]
        return Tensor._from_impl(CTensor([nf, count], result), None, None)

    def gather_cols(self, indices: list[int]) -> Tensor:
        flat = self.flat
        nf, ns = self.shape
        n = len(indices)
        result = [0.0] * (nf * n)
        for i in range(nf):
            src_row = i * ns
            dst_row = i * n
            for j, col in enumerate(indices):
                result[dst_row + j] = flat[src_row + col]
        return Tensor._from_impl(CTensor([nf, n], result), None, None)

    # Dot product
    def __matmul__(
            self, 
            other: Tensor
            ) -> Tensor:

        return Tensor._from_impl(
            self._impl.dot_tensor(other._impl),
            Operation.DOT,
            [self, other],
        )
