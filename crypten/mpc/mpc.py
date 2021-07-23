#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from functools import wraps

import torch
from torch import Tensor
from crypten import communicator as comm
from crypten.common.tensor_types import is_tensor
import crypten.common.approximations as approximations
from crypten.common.util import (
    ConfigBase,
    adaptive_pool2d_helper,
    pool2d_reshape,
    torch_cat,
    torch_stack,
)
from crypten.cuda import CUDALongTensor

from ..cryptensor import CrypTensor
from ..encoder import FixedPointEncoder
from .max_helper import _argmax_helper, _max_helper_all_tree_reductions
from .primitives.binary import BinarySharedTensor
from .primitives.converters import convert
from .ptype import ptype as Ptype


def mode(ptype, inplace=False):
    if inplace:

        def function_wrapper(func):
            # @wraps ensures docstrings are updated
            @wraps(func)
            def convert_wrapper(self, *args, **kwargs):
                self._tensor = convert(self._tensor, ptype)
                self.ptype = ptype
                self = func(self, *args, **kwargs)
                return self

            return convert_wrapper

    else:

        def function_wrapper(func):
            # @wraps ensures docstrings are updated
            @wraps(func)
            def convert_wrapper(self, *args, **kwargs):
                result = self.to(ptype)
                return func(result, *args, **kwargs)

            return convert_wrapper

    return function_wrapper


def _one_hot_to_index(tensor, dim, keepdim, device=None):
    """
    Converts a one-hot tensor output from an argmax / argmin function to a
    tensor containing indices from the input tensor from which the result of the
    argmax / argmin was obtained.
    """
    if dim is None:
        result = tensor.flatten()
        result = result * torch.tensor(list(range(tensor.nelement())), device=device)
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        result = tensor * torch.tensor(
            list(range(tensor.size(dim))), device=device
        ).view(size)
        return result.sum(dim, keepdim=keepdim)


@dataclass
class MPCConfig:
    """
    A configuration object for use by the MPCTensor.
    """

    # Used by max / argmax / min / argmin
    max_method: str = "log_reduction"

    # Used (for the moment) when generating the Beaver Triples
    active_security: bool = False


# Global config
config = MPCConfig()


class ConfigManager(ConfigBase):
    r"""
    Use this to temporarily change a value in the `mpc.config` object. The
    following sets `config.exp_iterations` to `10` for one function
    invocation and then sets it back to the previous value::

        with ConfigManager("exp_iterations", 10):
            tensor.exp()

    """

    def __init__(self, *args):
        super().__init__(config, *args)


class MPCTensor(CrypTensor):
    def __new__(cls, tensor, ptype=Ptype.arithmetic, device=None, *args, meta_size, **kwargs):
        """
        Creates the shared tensor from the input `tensor` provided by party `src`.
        The `ptype` defines the type of sharing used (default: arithmetic).

        The other parties can specify a `tensor` or `size` to determine the size
        of the shared tensor object to create. In this case, all parties must
        specify the same (tensor) size to prevent the party's shares from varying
        in size, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct size. The parties who do not know the
        tensor size beforehand can provide an empty tensor as input. This is
        guaranteed to produce correct behavior but requires an additional
        communication round.

        The parties can also set the `precision` and `device` for their share of
        the tensor. If `device` is unspecified, it is set to `tensor.device`.
        """
        if tensor is None:
            raise ValueError("Cannot initialize tensor with None.")

        # take required_grad from kwargs, input tensor, or set to False:
        default = tensor.requires_grad if torch.is_tensor(tensor) else False
        requires_grad = kwargs.pop("requires_grad", default)

        # call CrypTensor constructor:
        meta = torch.empty(meta_size, device='meta')
        self = cls._make_subclass(cls, meta, requires_grad)

        if device is None and hasattr(tensor, "device"):
            device = tensor.device

        assert str(device) != 'meta'

        # create the MPCTensor:
        tensor_type = ptype.to_tensor()
        if tensor == []:
            self._tensor = None  # catch it
        else:
            self._tensor = tensor_type(tensor=tensor, device=device, *args, **kwargs)
            assert self._tensor.size() == self.size(), f"{self._tensor.size()} != {self.size()}"
        self.ptype = ptype

        return self

    @property
    def _tensor(self):
        return self.__tensor

    @_tensor.setter
    def _tensor(self, x):
        self.__tensor = x
        if x is not None:
            assert self.size() == x.size(), f"{self.size()} != {x.size()}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        name = func.__name__
        A = torch.ops.aten
        print(f"{name}(...)")
        if func is A.cat:
            return MPCTensor.cat(*args, **kwargs)
        elif func is A.ones_like:
            self, dtype, layout, device, requires_grad, memory_format = args
            # TODO
            return self.new(torch.ones_like(self.share), meta_size=self.size())
        elif func is A.detach:
            # TODO: wut
            return args[0].clone()
        elif len(args) > 0 and isinstance(args[0], MPCTensor) and hasattr(args[0], f'dispatch_{name}'):
            return getattr(args[0], f"dispatch_{name}")(*args[1:], **kwargs)
        elif len(args) > 0 and isinstance(args[0], MPCTensor) and hasattr(args[0], name) and getattr(args[0], name) is not getattr(torch.Tensor, name):
            return getattr(args[0], f"{name}")(*args[1:], **kwargs)
        else:
            raise NotImplementedError(f"{func.__name__} not supported")

    @staticmethod
    def new(*args, **kwargs):
        """
        Creates a new MPCTensor, passing all args and kwargs into the constructor.
        """
        return MPCTensor(*args, **kwargs)

    @staticmethod
    def from_shares(share, precision=None, ptype=Ptype.arithmetic):
        result = MPCTensor([], meta_size=share.size())
        from_shares = ptype.to_tensor().from_shares
        result._tensor = from_shares(share, precision=precision)
        result.ptype = ptype
        return result

    def dispatch_clone(self):
        """Create a deep copy of the input tensor."""
        # TODO: Rename this to __deepcopy__()?
        result = MPCTensor([], meta_size=self.size())
        result._tensor = self._tensor.clone()
        result.ptype = self.ptype
        return result

    def shallow_copy(self):
        """Create a shallow copy of the input tensor."""
        # TODO: Rename this to __copy__()?
        result = MPCTensor([], meta_size=self.size())
        result._tensor = self._tensor
        result.ptype = self.ptype
        return result

    def dispatch_copy_(self, other):
        """Copies value of other MPCTensor into this MPCTensor."""
        assert isinstance(other, MPCTensor), "other must be MPCTensor"
        self._tensor.copy_(other._tensor)
        self.ptype = other.ptype

    def to(self, *args, **kwargs):
        r"""
        Depending on the input arguments,
        converts underlying share to the given ptype or
        performs `torch.to` on the underlying torch tensor

        To convert underlying share to the given ptype, call `to` as:
            to(ptype, **kwargs)

        It will call MPCTensor.to_ptype with the arguments provided above.

        Otherwise, `to` performs `torch.to` on the underlying
        torch tensor. See
        https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to
        for a reference of the parameters that can be passed in.

        Args:
            ptype: Ptype.arithmetic or Ptype.binary.
        """
        if "ptype" in kwargs:
            return self._to_ptype(**kwargs)
        elif args and isinstance(args[0], Ptype):
            ptype = args[0]
            return self._to_ptype(ptype, **kwargs)
        else:
            share = self.share.to(*args, **kwargs)
            if share.is_cuda:
                share = CUDALongTensor(share)
            self.share = share
            return self

    def _to_ptype(self, ptype, **kwargs):
        r"""
        Convert MPCTensor's underlying share to the corresponding ptype
        (ArithmeticSharedTensor, BinarySharedTensor)

        Args:
            ptype (Ptype.arithmetic or Ptype.binary): The ptype to convert
                the shares to.
            precision (int, optional): Precision of the fixed point encoder when
                converting a binary share to an arithmetic share. It will be ignored
                if the ptype doesn't match.
            bits (int, optional): If specified, will only preserve the bottom `bits` bits
                of a binary tensor when converting from a binary share to an arithmetic share.
                It will be ignored if the ptype doesn't match.
        """
        retval = self.clone()
        if retval.ptype == ptype:
            return retval
        retval._tensor = convert(self._tensor, ptype, **kwargs)
        retval.ptype = ptype
        return retval

    def arithmetic(self):
        """Converts self._tensor to arithmetic secret sharing"""
        return self.to(Ptype.arithmetic)

    def binary(self):
        """Converts self._tensor to binary secret sharing"""
        return self.to(Ptype.binary)

    @property
    def device(self):
        """Return the `torch.device` of the underlying share"""
        return self.share.device

    @property
    def is_cuda(self):
        """Return True if the underlying share is stored on GPU, False otherwise"""
        return self.share.is_cuda

    def cuda(self, *args, **kwargs):
        """Call `torch.Tensor.cuda` on the underlying share"""
        self.share = CUDALongTensor(self.share.cuda(*args, **kwargs))
        return self

    def cpu(self):
        """Call `torch.Tensor.cpu` on the underlying share"""
        self.share = self.share.cpu()
        return self

    def get_plain_text(self, dst=None):
        """Decrypts the tensor."""
        return self._tensor.get_plain_text(dst=dst)

    def reveal(self, dst=None):
        """Decrypts the tensor without any downscaling."""
        return self._tensor.reveal(dst=dst)

    def __repr__(self):
        """Returns a representation of the tensor useful for debugging."""
        from crypten.debug import debug_mode

        share = self.share
        plain_text = self._tensor.get_plain_text() if debug_mode() else "HIDDEN"
        ptype = self.ptype
        return (
            f"MPCTensor(\n\t_tensor={share}\n"
            f"\tplain_text={plain_text}\n\tptype={ptype}\n)"
        )

    def __hash__(self):
        return hash(self.share)

    @property
    def share(self):
        """Returns underlying share"""
        return self._tensor.share

    @share.setter
    def share(self, value):
        """Sets share to value"""
        self._tensor.share = value

    @property
    def data(self):
        """Returns share data"""
        return self.share.data

    @data.setter
    def data(self, value):
        """Sets data to value"""
        self.share.data = value

    @property
    def encoder(self):
        """Returns underlying encoder"""
        return self._tensor.encoder

    @encoder.setter
    def encoder(self, value):
        """Sets encoder to value"""
        self._tensor.encoder = value

    @staticmethod
    def __cat_stack_helper(op, tensors, *args, **kwargs):
        assert op in ["cat", "stack"], "Unsupported op for helper function"
        assert isinstance(tensors, list), "%s input must be a list" % op
        assert len(tensors) > 0, "expected a non-empty list of MPCTensors"

        _ptype = kwargs.pop("ptype", None)
        # Populate ptype field
        if _ptype is None:
            for tensor in tensors:
                if isinstance(tensor, MPCTensor):
                    _ptype = tensor.ptype
                    break
        if _ptype is None:
            _ptype = Ptype.arithmetic

        # Make all inputs MPCTensors of given ptype
        for i, tensor in enumerate(tensors):
            if tensor.ptype != _ptype:
                tensors[i] = tensor.to(_ptype)

        # Operate on all input tensors
        result = tensors[0].clone()
        funcs = {"cat": torch_cat, "stack": torch_stack}
        result.share = funcs[op]([tensor.share for tensor in tensors], *args, **kwargs)
        return result

    @staticmethod
    def cat(tensors, *args, **kwargs):
        """Perform matrix concatenation"""
        return MPCTensor.__cat_stack_helper("cat", tensors, *args, **kwargs)

    @staticmethod
    def stack(tensors, *args, **kwargs):
        """Perform tensor stacking"""
        return MPCTensor.__cat_stack_helper("stack", tensors, *args, **kwargs)

    @staticmethod
    def rand(*sizes, device=None):
        """
        Returns a tensor with elements uniformly sampled in [0, 1). The uniform
        random samples are generated by generating random bits using fixed-point
        encoding and converting the result to an ArithmeticSharedTensor.
        """
        rand = MPCTensor([])
        encoder = FixedPointEncoder()
        rand._tensor = BinarySharedTensor.rand(
            *sizes, bits=encoder._precision_bits, device=device
        )
        rand._tensor.encoder = encoder
        rand.ptype = Ptype.binary
        return rand.to(Ptype.arithmetic, bits=encoder._precision_bits)

    @staticmethod
    def randn(*sizes, device=None):
        """
        Returns a tensor with normally distributed elements. Samples are
        generated using the Box-Muller transform with optimizations for
        numerical precision and MPC efficiency.
        """
        u = MPCTensor.rand(*sizes, device=device).flatten()
        odd_numel = u.numel() % 2 == 1
        if odd_numel:
            u = MPCTensor.cat([u, MPCTensor.rand((1,), device=device)])

        n = u.numel() // 2
        u1 = u[:n]
        u2 = u[n:]

        # Radius = sqrt(- 2 * log(u1))
        r2 = -2 * u1.log(input_in_01=True)
        r = r2.sqrt()

        # Theta = cos(2 * pi * u2) or sin(2 * pi * u2)
        cos, sin = u2.sub(0.5).mul(6.28318531).cossin()

        # Generating 2 independent normal random variables using
        x = r.mul(sin)
        y = r.mul(cos)
        z = MPCTensor.cat([x, y])

        if odd_numel:
            z = z[1:]

        return z.view(*sizes)

    def bernoulli(self):
        """Returns a tensor with elements in {0, 1}. The i-th element of the
        output will be 1 with probability according to the i-th value of the
        input tensor."""
        return self > MPCTensor.rand(self.size(), device=self.device)

    # Comparators
    @mode(Ptype.binary)
    def _ltz(self, _scale=True):
        """Returns 1 for elements that are < 0 and 0 otherwise"""
        shift = torch.iinfo(torch.long).bits - 1
        result = (self >> shift).to(Ptype.arithmetic, bits=1)
        if _scale:
            return result * result.encoder._scale
        else:
            result.encoder._scale = 1
            return result

    @mode(Ptype.arithmetic)
    def ge(self, y, _scale=True):
        """Returns self >= y"""
        return 1 - self.lt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def gt(self, y, _scale=True):
        """Returns self > y"""
        return (-self + y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def le(self, y, _scale=True):
        """Returns self <= y"""
        return 1 - self.gt(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def lt(self, y, _scale=True):
        """Returns self < y"""
        return (self - y)._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def eq(self, y, _scale=True):
        """Returns self == y"""
        if comm.get().get_world_size() == 2:
            return (self - y)._eqz_2PC(_scale=_scale)

        return 1 - self.ne(y, _scale=_scale)

    @mode(Ptype.arithmetic)
    def ne(self, y, _scale=True):
        """Returns self != y"""
        if comm.get().get_world_size() == 2:
            return 1 - self.eq(y, _scale=_scale)

        difference = self - y
        difference.share = torch_stack([difference.share, -(difference.share)])
        return difference._ltz(_scale=_scale).sum(0)

    @mode(Ptype.arithmetic)
    def _eqz_2PC(self, _scale=True):
        """Returns self == 0"""
        # Create BinarySharedTensors from shares
        x0 = MPCTensor(self.share, src=0, ptype=Ptype.binary)
        x1 = MPCTensor(-self.share, src=1, ptype=Ptype.binary)

        # Perform equality testing using binary shares
        x0._tensor = x0._tensor.eq(x1._tensor)
        x0.encoder = x0.encoder if _scale else self.encoder

        # Convert to Arithmetic sharing
        result = x0.to(Ptype.arithmetic, bits=1)

        if not _scale:
            result.encoder._scale = 1

        return result

    @mode(Ptype.arithmetic)
    def sign(self, _scale=True):
        """Computes the sign value of a tensor (0 is considered positive)"""
        return 1 - 2 * self._ltz(_scale=_scale)

    @mode(Ptype.arithmetic)
    def abs(self):
        """Computes the absolute value of a tensor"""
        return self * self.sign(_scale=False)

    @mode(Ptype.arithmetic)
    def relu(self):
        """Compute a Rectified Linear function on the input tensor."""
        return self * self.ge(0, _scale=False)

    @mode(Ptype.arithmetic)
    def weighted_index(self, dim=None):
        """
        Returns a tensor with entries that are one-hot along dimension `dim`.
        These one-hot entries are set at random with weights given by the input
        `self`.

        Examples::

            >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
            >>> index = encrypted_tensor.weighted_index().get_plain_text()
            # With 1 / 7 probability
            torch.tensor([1., 0.])

            # With 6 / 7 probability
            torch.tensor([0., 1.])
        """
        if dim is None:
            return self.flatten().weighted_index(dim=0).view(self.size())

        x = self.cumsum(dim)
        max_weight = x.index_select(
            dim, torch.tensor(x.size(dim) - 1, device=self.device)
        )
        r = MPCTensor.rand(max_weight.size(), device=self.device) * max_weight

        gt = x.gt(r, _scale=False)
        shifted = gt.roll(1, dims=dim)
        shifted.share.index_fill_(dim, torch.tensor(0, device=self.device), 0)

        return gt - shifted

    @mode(Ptype.arithmetic)
    def weighted_sample(self, dim=None):
        """
        Samples a single value across dimension `dim` with weights corresponding
        to the values in `self`

        Returns the sample and the one-hot index of the sample.

        Examples::

            >>> encrypted_tensor = MPCTensor(torch.tensor([1., 6.]))
            >>> index = encrypted_tensor.weighted_sample().get_plain_text()
            # With 1 / 7 probability
            (torch.tensor([1., 0.]), torch.tensor([1., 0.]))

            # With 6 / 7 probability
            (torch.tensor([0., 6.]), torch.tensor([0., 1.]))
        """
        indices = self.weighted_index(dim)
        sample = self.mul(indices).sum(dim)
        return sample, indices

    # max / min-related functions
    @mode(Ptype.arithmetic)
    def argmax(self, dim=None, keepdim=False, one_hot=True):
        """Returns the indices of the maximum value of all elements in the
        `input` tensor.
        """
        # TODO: Make dim an arg.
        if self.dim() == 0:
            result = (
                MPCTensor(torch.ones((), device=self.device))
                if one_hot
                else MPCTensor(torch.zeros((), device=self.device))
            )
            return result

        result = _argmax_helper(
            self, dim, one_hot, config.max_method, _return_max=False
        )

        if not one_hot:
            result = _one_hot_to_index(result, dim, keepdim, self.device)
        return result

    @mode(Ptype.arithmetic)
    def argmin(self, dim=None, keepdim=False, one_hot=True):
        """Returns the indices of the minimum value of all elements in the
        `input` tensor.
        """
        # TODO: Make dim an arg.
        return (-self).argmax(dim=dim, keepdim=keepdim, one_hot=one_hot)

    @mode(Ptype.arithmetic)
    def max(self, dim=None, keepdim=False, one_hot=True):
        """Returns the maximum value of all elements in the input tensor."""
        # TODO: Make dim an arg.
        method = config.max_method
        if dim is None:
            if method in ["log_reduction", "double_log_reduction"]:
                # max_result can be obtained directly
                max_result = _max_helper_all_tree_reductions(self, method=method)
            else:
                # max_result needs to be obtained through argmax
                with ConfigManager("max_method", method):
                    argmax_result = self.argmax(one_hot=True)
                max_result = self.mul(argmax_result).sum()
            return max_result
        else:
            argmax_result, max_result = _argmax_helper(
                self, dim=dim, one_hot=True, method=method, _return_max=True
            )
            if max_result is None:
                max_result = (self * argmax_result).sum(dim=dim, keepdim=keepdim)
            if keepdim:
                max_result = (
                    max_result.unsqueeze(dim)
                    if max_result.dim() < self.dim()
                    else max_result
                )
            if one_hot:
                return max_result, argmax_result
            else:
                return (
                    max_result,
                    _one_hot_to_index(argmax_result, dim, keepdim, self.device),
                )

    @mode(Ptype.arithmetic)
    def min(self, dim=None, keepdim=False, one_hot=True):
        """Returns the minimum value of all elements in the input tensor."""
        # TODO: Make dim an arg.
        result = (-self).max(dim=dim, keepdim=keepdim, one_hot=one_hot)
        if dim is None:
            return -result
        else:
            return -result[0], result[1]

    @mode(Ptype.arithmetic)
    def max_pool2d(
        self,
        kernel_size,
        padding=0,
        stride=None,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        """Applies a 2D max pooling over an input signal composed of several
        input planes.
        """
        max_input = self.shallow_copy()
        max_input.share, output_size = pool2d_reshape(
            self.share,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
            # padding with extremely negative values to avoid choosing pads.
            # The magnitude of this value should not be too large because
            # multiplication can otherwise fail.
            pad_value=(-(2 ** 24)),
            # TODO: Find a better solution for padding with max_pooling
        )
        max_vals, argmax_vals = max_input.max(dim=-1, one_hot=True)
        max_vals = max_vals.view(output_size)
        if return_indices:
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            argmax_vals = argmax_vals.view(output_size + kernel_size)
            return max_vals, argmax_vals
        return max_vals

    @mode(Ptype.arithmetic)
    def _max_pool2d_backward(
        self,
        indices,
        kernel_size,
        padding=None,
        stride=None,
        dilation=1,
        ceil_mode=False,
        output_size=None,
    ):
        """Implements the backwards for a `max_pool2d` call."""
        # Setup padding
        if padding is None:
            padding = 0
        if isinstance(padding, int):
            padding = padding, padding
        assert isinstance(padding, tuple), "padding must be a int, tuple, or None"
        p0, p1 = padding

        # Setup stride
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = stride, stride
        assert isinstance(stride, tuple), "stride must be a int, tuple, or None"
        s0, s1 = stride

        # Setup dilation
        if isinstance(stride, int):
            dilation = dilation, dilation
        assert isinstance(dilation, tuple), "dilation must be a int, tuple, or None"
        d0, d1 = dilation

        # Setup kernel_size
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        assert isinstance(padding, tuple), "padding must be a int or tuple"
        k0, k1 = kernel_size

        assert self.dim() == 4, "Input to _max_pool2d_backward must have 4 dimensions"
        assert (
            indices.dim() == 6
        ), "Indices input for _max_pool2d_backward must have 6 dimensions"

        # Computes one-hot gradient blocks from each output variable that
        # has non-zero value corresponding to the argmax of the corresponding
        # block of the max_pool2d input.
        kernels = self.view(self.size() + (1, 1)) * indices

        # Use minimal size if output_size is not specified.
        if output_size is None:
            output_size = (
                self.size(0),
                self.size(1),
                s0 * self.size(2) - 2 * p0,
                s1 * self.size(3) - 2 * p1,
            )

        # Account for input padding
        result_size = list(output_size)
        result_size[-2] += 2 * p0
        result_size[-1] += 2 * p1

        # Account for input padding implied by ceil_mode
        if ceil_mode:
            c0 = self.size(-1) * s1 + (k1 - 1) * d1 - output_size[-1]
            c1 = self.size(-2) * s0 + (k0 - 1) * d0 - output_size[-2]
            result_size[-2] += c0
            result_size[-1] += c1

        # Sum the one-hot gradient blocks at corresponding index locations.
        result = MPCTensor(torch.zeros(result_size, device=kernels.device))
        for i in range(self.size(2)):
            for j in range(self.size(3)):
                left_ind = s0 * i
                top_ind = s1 * j

                result[
                    :,
                    :,
                    left_ind : left_ind + k0 * d0 : d0,
                    top_ind : top_ind + k1 * d1 : d1,
                ] += kernels[:, :, i, j]

        # Remove input padding
        if ceil_mode:
            result = result[:, :, : result.size(2) - c0, : result.size(3) - c1]
        result = result[:, :, p0 : result.size(2) - p0, p1 : result.size(3) - p1]

        return result

    def adaptive_avg_pool2d(self, output_size):
        r"""
        Applies a 2D adaptive average pooling over an input signal composed of
        several input planes.

        See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

        Args:
            output_size: the target output size (single integer or
                double-integer tuple)
        """
        resized_input, args, kwargs = adaptive_pool2d_helper(
            self, output_size, reduction="mean"
        )
        return resized_input.avg_pool2d(*args, **kwargs)

    def adaptive_max_pool2d(self, output_size, return_indices=False):
        r"""Applies a 2D adaptive max pooling over an input signal composed of
        several input planes.

        See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

        Args:
            output_size: the target output size (single integer or
                double-integer tuple)
            return_indices: whether to return pooling indices. Default: ``False``
        """
        resized_input, args, kwargs = adaptive_pool2d_helper(
            self, output_size, reduction="max"
        )
        return resized_input.max_pool2d(*args, **kwargs, return_indices=return_indices)

    def where(self, condition, y):
        """Selects elements from self or y based on condition

        Args:
            condition (torch.bool or MPCTensor): when True yield self,
                otherwise yield y
            y (torch.tensor or MPCTensor): values selected at indices
                where condition is False.

        Returns: MPCTensor or torch.tensor
        """
        if is_tensor(condition):
            condition = condition.float()
            y_masked = y * (1 - condition)
        else:
            # encrypted tensor must be first operand
            y_masked = (1 - condition) * y

        return self * condition + y_masked

    @mode(Ptype.arithmetic)
    def div(self, y):
        r"""Divides each element of :attr:`self` with the scalar :attr:`y` or
        each element of the tensor :attr:`y` and returns a new resulting tensor.

        For `y` a scalar:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}}

        For `y` a tensor:

        .. math::
            \text{out}_i = \frac{\text{self}_i}{\text{y}_i}

        Note for :attr:`y` a tensor, the shapes of :attr:`self` and :attr:`y` must be
        `broadcastable`_.

        .. _broadcastable:
            https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics"""  # noqa: B950
        result = self.clone()
        if isinstance(y, CrypTensor):
            result.share = torch.broadcast_tensors(result.share, y.share)[0].clone()
        elif is_tensor(y):
            result.share = torch.broadcast_tensors(result.share, y)[0].clone()

        if isinstance(y, MPCTensor):
            return result.mul(y.reciprocal())
        result._tensor.div_(y)
        return result

    def div_(self, y):
        """In-place version of :meth:`div`"""
        if isinstance(y, MPCTensor):
            return self.mul_(y.reciprocal())
        self._tensor.div_(y)
        return self

    def index_add(self, dim, index, tensor):
        """Performs out-of-place index_add: Accumulate the elements of tensor into the
        self tensor by adding to the indices in the order given in index.
        """
        result = self.clone()
        assert index.dim() == 1, "index needs to be a vector"
        public = isinstance(tensor, (int, float)) or is_tensor(tensor)
        private = isinstance(tensor, MPCTensor)
        if public:
            result._tensor.index_add_(dim, index, tensor)
        elif private:
            result._tensor.index_add_(dim, index, tensor._tensor)
        else:
            raise TypeError("index_add second tensor of unsupported type")
        return result

    def dispatch_convolution_overrideable(self, *args, **kwargs):
        weight, bias, stride, padding, dilation, transposed, output_padding, groups = args
        assert bias is None
        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        assert transposed is False
        assert output_padding == [0, 0], output_padding
        return self.dispatch_conv2d(weight, **kwargs)

    def scatter_add(self, dim, index, other):
        """Adds all values from the tensor other into self at the indices
        specified in the index tensor.
        """
        result = self.clone()
        public = isinstance(other, (int, float)) or is_tensor(other)
        private = isinstance(other, CrypTensor)
        if public:
            result._tensor.scatter_add_(dim, index, other)
        elif private:
            result._tensor.scatter_add_(dim, index, other._tensor)
        else:
            raise TypeError("scatter_add second tensor of unsupported type")
        return result

    def scatter(self, dim, index, src):
        """Out-of-place version of :meth:`MPCTensor.scatter_`"""
        result = self.clone()
        if is_tensor(src):
            src = MPCTensor(src)
        assert isinstance(src, MPCTensor), "Unrecognized scatter src type: %s" % type(
            src
        )
        result.share.scatter_(dim, index, src.share)
        return result

    def unbind(self, dim=0):
        shares = self.share.unbind(dim=dim)
        results = tuple(
            MPCTensor(0, ptype=self.ptype, device=self.device)
            for _ in range(len(shares))
        )
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def split(self, split_size, dim=0):
        shares = self.share.split(split_size, dim=dim)
        results = tuple(
            MPCTensor(0, meta_size=s.size(), ptype=self.ptype, device=self.device)
            for s in shares
        )
        for i in range(len(shares)):
            results[i].share = shares[i]
        return results

    def set(self, enc_tensor):
        """
        Sets self encrypted to enc_tensor in place by setting
        shares of self to those of enc_tensor.

        Args:
            enc_tensor (MPCTensor): with encrypted shares.
        """
        if is_tensor(enc_tensor):
            enc_tensor = MPCTensor(enc_tensor)
        assert isinstance(enc_tensor, MPCTensor), "enc_tensor must be an MPCTensor"
        self.share.set_(enc_tensor.share)
        return self


OOP_UNARY_FUNCTIONS = {
    "avg_pool2d": Ptype.arithmetic,
    "take": Ptype.arithmetic,
    "square": Ptype.arithmetic,
    "prod": Ptype.arithmetic,
    "mean": Ptype.arithmetic,
    "var": Ptype.arithmetic,
    "neg": Ptype.arithmetic,
    "__neg__": Ptype.arithmetic,
    "invert": Ptype.binary,
    "lshift": Ptype.binary,
    "rshift": Ptype.binary,
    "__invert__": Ptype.binary,
    "__lshift__": Ptype.binary,
    "__rshift__": Ptype.binary,
    "__rand__": Ptype.binary,
    "__rxor__": Ptype.binary,
    "__ror__": Ptype.binary,
}

OOP_BINARY_FUNCTIONS = {
    "add": Ptype.arithmetic,
    "sub": Ptype.arithmetic,
    "mul": Ptype.arithmetic,
    "matmul": Ptype.arithmetic,
    "conv1d": Ptype.arithmetic,
    "conv2d": Ptype.arithmetic,
    "conv_transpose1d": Ptype.arithmetic,
    "conv_transpose2d": Ptype.arithmetic,
    "dot": Ptype.arithmetic,
    "ger": Ptype.arithmetic,
    "__xor__": Ptype.binary,
    "__or__": Ptype.binary,
    "__and__": Ptype.binary,
}

INPLACE_UNARY_FUNCTIONS = {
    "neg_": Ptype.arithmetic,
    "invert_": Ptype.binary,
    "lshift_": Ptype.binary,
    "rshift_": Ptype.binary,
}

INPLACE_BINARY_FUNCTIONS = {
    "add_": Ptype.arithmetic,
    "sub_": Ptype.arithmetic,
    "mul_": Ptype.arithmetic,
    "__ior__": Ptype.binary,
    "__ixor__": Ptype.binary,
    "__iand__": Ptype.binary,
}


def _add_oop_unary_passthrough_function(name, preferred=None):
    def ou_wrapper_function(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = getattr(result._tensor, name)(*args, **kwargs)
        return result

    if preferred is None:
        setattr(MPCTensor, f'dispatch_{name}', ou_wrapper_function)
    else:
        setattr(MPCTensor, f'dispatch_{name}', mode(preferred, False)(ou_wrapper_function))


def _add_oop_binary_passthrough_function(name, preferred=None):
    def ob_wrapper_function(self, value, *args, **kwargs):
        result = self.shallow_copy()
        if isinstance(value, MPCTensor):
            value = value._tensor
        result_tensor = getattr(result._tensor, name)(value, *args, **kwargs)
        result2 = MPCTensor([], meta_size=result_tensor.size())
        result2._tensor = result_tensor
        return result2

    if preferred is None:
        setattr(MPCTensor, f'dispatch_{name}', ob_wrapper_function)
    else:
        setattr(MPCTensor, f'dispatch_{name}', mode(preferred, False)(ob_wrapper_function))


def _add_inplace_unary_passthrough_function(name, preferred=None):
    def iu_wrapper_function(self, *args, **kwargs):
        self._tensor = getattr(self._tensor, name)(*args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, f'dispatch_{name}', iu_wrapper_function)
    else:
        setattr(MPCTensor, f'dispatch_{name}', mode(preferred, True)(iu_wrapper_function))


def _add_inplace_binary_passthrough_function(name, preferred=None):
    def ib_wrapper_function(self, value, *args, **kwargs):
        if isinstance(value, MPCTensor):
            value = value._tensor
        self._tensor = getattr(self._tensor, name)(value, *args, **kwargs)
        return self

    if preferred is None:
        setattr(MPCTensor, f'dispatch_{name}', ib_wrapper_function)
    else:
        setattr(MPCTensor, f'dispatch_{name}', mode(preferred, True)(ib_wrapper_function))


for func_name, preferred_type in OOP_UNARY_FUNCTIONS.items():
    _add_oop_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in OOP_BINARY_FUNCTIONS.items():
    _add_oop_binary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_UNARY_FUNCTIONS.items():
    _add_inplace_unary_passthrough_function(func_name, preferred_type)

for func_name, preferred_type in INPLACE_BINARY_FUNCTIONS.items():
    _add_inplace_binary_passthrough_function(func_name, preferred_type)
