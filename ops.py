from torch.autograd import Function


class BinaryQuantization(Function):
    """ Implementation of paper: Deep Learning With Low Precision by Half-Wave Gaussian Quantization
    Forward using sign()
    Backward using hard-tanh()
    """
    @staticmethod
    def forward(ctx, weights):
        ctx.save_for_backward(weights)
        mask = (weights > 0).float()
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_variables
        tanh_gradient = (weights.abs() <= 1).float()
        weights_gradient = grad_output * tanh_gradient
        return weights_gradient


binary_quantization = BinaryQuantization.apply
