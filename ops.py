import torch
from torch.autograd import Function


class BinaryQuantization(Function):
    """ Implementation of paper: Deep Learning With Low Precision by Half-Wave Gaussian Quantization
    Forward using sign()
    Backward using hard-tanh()
    """
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mask = (torch.sign(weights)+1)/2
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_variables
        tanh_gradient = (weights.abs() <= 1).float()
        # TODO: what is the multiplier here
        weights_gradient = grad_output * tanh_gradient
        return weights_gradient


class DNS(Function):
    """ Implementation of paper: Dynamic network surgery for efficient dnns
    Forward with dotproducting mask
    Backward for all weights
    """
    @staticmethod
    def forward(ctx, weights, mask):
        return weights * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


binary_quantization = BinaryQuantization.apply
dns = DNS.apply
