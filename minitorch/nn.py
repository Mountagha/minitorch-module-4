from typing import Tuple

import numpy as np

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    # make it contiguous to be able to call view.
    input = input.contiguous()
    # splitting height and width by kh and kw respectively.
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)  # moving kh and kw at the end
    input = input.contiguous()  # so that we can apply view again.
    return (
        input.view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    tile_tensor, new_height, new_width = tile(input, kernel)
    avgpool_tensor = tile_tensor.mean(4)
    return avgpool_tensor.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError("Need to implement for Task 4.4")
        dims = dim.to_numpy().astype(int).tolist()
        max_tensor = max_reduce(input, dims[0])
        ctx.save_for_backward(input, max_tensor)
        return max_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError("Need to implement for Task 4.4")
        input, max_tensor = ctx.saved_values
        return grad_output * (input == max_tensor), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError("Need to implement for Task 4.4")
    # shift the input tensor by substracting the max value for numerical stability.
    flatten_shape = int(np.prod(input.shape))
    shifted_input = input - max(
        input.contiguous().view(
            flatten_shape,
        ),
        0,
    )

    # compute the exponentiels values
    exp_input = shifted_input.exp()

    # compute the sum of exponentiels along the specified dimension
    sum_exp_input = exp_input.sum(dim)

    # normalize by dividing by the exponentiels by the sum of exponentiels
    softmax_input = exp_input / sum_exp_input

    return softmax_input


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError("Need to implement for Task 4.4")
    # The provided link gives the trick to compute the log-sum-exp LSE as follows
    # LSE(x1,....xn) = x* + log(exp(x1-x*) + ... + expr(xn - x*))
    # where x* = max{x1,....,xn}

    # compute the max tensor
    max_tensor = max(input, dim)

    # substract the max from the input
    shifted_input = input - max_tensor

    # compute the exponentiels values
    exp_input = shifted_input.exp()

    # sum the exponential.
    sum_input = exp_input.sum(dim)

    # take the log.
    log_input = sum_input.log()

    # add the max
    log_sum_exp_input = log_input + max_tensor

    return input - log_sum_exp_input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    tiled_tensor, new_height, new_width = tile(input, kernel)
    max_pooled_tensor = max(tiled_tensor, 4)
    return max_pooled_tensor.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    # raise NotImplementedError("Need to implement for Task 4.4")
    if ignore:
        return input
    # create a mask tensor.
    mask = rand(input.shape) > rate
    # apply the filter to drop all values that correspond to false (0)
    dropout_tensor = input * mask
    return dropout_tensor
