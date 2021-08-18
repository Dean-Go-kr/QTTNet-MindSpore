import mindspore as ms
import mindspore.numpy as mnp
import numpy as np
import mindspore.nn as nn

bitsW = 8
bitsA = 8

"""
量化部分（自定义梯度计算）以后实现
"""

def scale(x):
    scale = ms.Tensor.max(ms.Tensor.abs(x))
    result = 2.**mnp.round(mnp.log2(scale))
    return result


def delta(bits):
    result = (2.**(1-bits))
    return result


def clip(x, bits):
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil  = 1 - step
    floor = step - 1
    result = np.clamp(x, floor, ceil)
    return result


def quant(x, bits):
    if bits >= 32:
        result = x
    else:
        result = mnp.round(x/delta(bits))*delta(bits)
    return result


def qw(x):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = np.clip(quant(x,bits),bits)
    return result


def qa(x):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        result = quant(x,bits)
    return result


class QW():
    def __init__(self):
        super(QW, self).__init__()

    def construct(self, x):
        result = qw(x)
        return result

    def bprop(self, grad_output):
        grad_input = grad_output
        return grad_input
quantizeW = QW()


class QA():
    def construct(self, x):
        self.save_for_backward(x)
        result = qa(x)
        return result

    def bprop(self, grad_output):
        grad_input = grad_output
        return grad_input
quantizeAE = QA.apply


# import numpy as np
# np.random.seed(10)
# shape = (5,5)
# test_data = np.random.rand(*shape)
# test_tensor = torch.from_numpy(test_data).float()
# result = qg(test_tensor)
# print(test_tensor)
# print(result)