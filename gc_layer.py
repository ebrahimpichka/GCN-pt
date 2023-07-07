import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn. modules.module import Module
from torch.nn import init
import torch.nn.functional as F
import math

class GraphConvolution(Module):
    r"""Applies a graph convolution transformation to the input feature tensor and sparse adjacency matrix.
          based on https://arxiv.org/pdf/1609.02907.pdf

    Args:
        in_features: (int) size of each input sample
        out_features: (int) size of each output sample
        bias: (bool) If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight : (Parameter) the learnable weights of the module of shape (in_features x out_features).
        bias : (Parameter) the learnable bias of the module of shape (out_features}).
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, adj_mat: Tensor) -> Tensor:
        '''
        computes ÂH⁽ˡ⁾W⁽ˡ⁾ where Â = D⁻¹/²ÃD⁻¹/² + I is the pre-computed adjacency matrix with self-connections added

        Args:
            input: (Tensor) input node feature tensor of shape (batch_size, in_features)
            adj_mat: (Tensor) sparse adjacency matrix of the undirected graph
              added self-connections, of shape (batch_size, batch_size)
              
        Returns:
            output: (Tensor) output node feature tensor of shape (batch_size, out_features)   
        '''
        support = torch.mm(input, self.weight.t())
        output = torch.spmm(adj_mat, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )