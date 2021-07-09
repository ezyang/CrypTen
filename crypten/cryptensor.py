#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch

from .common import approximations, regular
from .debug import validation_mode, validate_correctness
from .gradients import (
    AutogradContext,
    BaseAutogradContext,
    get_grad_fn,
    get_grad_fn_registry,
)


# list of all static functions that CrypTensors support:
STATIC_FUNCTIONS = ["cat", "stack"]
STATIC_FUNCTION_MAPPING = {getattr(torch, name): name for name in STATIC_FUNCTIONS}


def _find_all_cryptensors(inputs):
    """
    Recursively find all CrypTensors in an input list, tuple, set, or dict.
    """
    cryptensors = []
    for input in inputs:
        if isinstance(input, CrypTensor):
            cryptensors.append(input)
        elif isinstance(input, (list, tuple, set)):
            cryptensors.extend(_find_all_cryptensors(input))
        elif isinstance(input, dict):
            for value in input.values():
                cryptensors.extend(_find_all_cryptensors(value))
    return cryptensors


class CrypTensor(torch.Tensor):
    pass
    """
    Abstract implementation of encrypted tensor type. Every subclass of `CrypTensor`
    must implement the methods defined here. The actual tensor data should live in
    an instance attribute called `_tensor`. When implemented, the `CrypTensor`
    provides a full autograd implementation to the user.
    """

    # attributes that should be dispatched to underlying tensor:
    PROTECTED_ATTRIBUTES = [
        "__dict__",
        "__class__",
        "requires_grad",
        "grad",
        "grad_fn",
        "grad_expected",
        "grad_received",
        "children",
        "ctx",
        "backward",
        "detach",
        "detach_",
        "_reset_gradients",
    ]

    # functions that should be implemented by CrypTensor subclass:
    REQUIRED_FUNCTIONS = [
        "_ltz",
        "add",
        "avg_pool1d",
        "avg_pool2d",
        "clone",
        "conv1d",
        "conv2d",
        "copy_",
        "div_",
        "matmul",
        "neg",
    ]

    # dict for storing functional overrides from subclasses:
    FUNCTION_OVERRIDES = {}

    # mapping of Python built-in methods to CrypTensor methods:
    PYTHON_BUILTIN = {
        "__abs__": "abs",
        "__neg__": "neg",
        "__pow__": "pow",
        "__add__": "add",
        "__radd__": "add",
        "__sub__": "sub",
        "__rsub__": "__rsub__",
        "__mul__": "mul",
        "__rmul__": "mul",
        "__div__": "div",
        "__truediv__": "div",
        "__rtruediv__": "__rtruediv__",
        "__matmul__": "matmul",
        "__imatmul__": "matmul",  # not in-place, matching PyTorch
    }
    # TODO: Automatically register all these functions in CrypTensor?

    def __abs__(self):
        return self.abs()

    def __add__(self, tensor):
        """Adds tensor to this tensor."""
        return self.add(tensor)

    __radd__ = __add__

    def __iadd__(self, tensor):
        """Adds tensor to this tensor (in-place)."""
        return self.add_(tensor)

    def __sub__(self, tensor):
        """Subtracts tensor from this tensor."""
        return self.sub(tensor)

    def __rsub__(self, tensor):
        """Subtracts self from tensor."""
        return -self + tensor

    def __isub__(self, tensor):
        """Subtracts tensor from this tensor (in-place)."""
        return self.sub_(tensor)

    def __mul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul(tensor)

    __rmul__ = __mul__

    def __imul__(self, tensor):
        """Element-wise multiply with a tensor."""
        return self.mul_(tensor)

    def __div__(self, tensor):
        """Element-wise divide by a tensor."""
        return self.div(tensor)

    def __truediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div(scalar)

    def __itruediv__(self, scalar):
        """Element-wise divide by a tensor."""
        return self.div_(scalar)

    def __neg__(self):
        return self.neg()

    def __matmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        return self.matmul(tensor)

    def __imatmul__(self, tensor):
        """Perform matrix multiplication using some tensor"""
        # Note: Matching PyTorch convention, which is not in-place here.
        return self.matmul(tensor)

    def square(self):
        """
        Computes the square of :attr:`self`
        """
        return self * self

    def __le__(self, tensor):
        """Element-wise less than or equal to"""
        return self.le(tensor)

    def __lt__(self, tensor):
        """Element-wise less than"""
        return self.lt(tensor)

    def __bool__(self):
        """Override bool operator since encrypted tensors cannot evaluate"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")

    def __nonzero__(self):
        """__bool__ for backwards compatibility with Python 2"""
        raise RuntimeError("Cannot evaluate CrypTensors to boolean values")


# Register function approximations
for func in approximations.__all__:
    setattr(CrypTensor, func, getattr(approximations, func))
