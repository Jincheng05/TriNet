import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import math
from torch.nn.parameter import Parameter

def _triple(x):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return (x, x, x)

class CondConv3d(nn.Module):
    """ Conditional Convolutional Layer for 3D inputs (as previously defined) """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'num_experts']

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', num_experts=4):
        super().__init__()

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if padding_mode not in ['zeros', 'reflect', 'replicate', 'circular']:
            raise ValueError("padding_mode can only be 'zeros', 'reflect', 'replicate', 'circular', but got {}".format(padding_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.num_experts = num_experts

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weights):
        B, C_in, D, H, W = x.shape
        B_route, N_experts = routing_weights.shape

        if B != B_route:
            raise ValueError(f"Batch sizes of input ({B}) and routing_weights ({B_route}) must match.")
        if N_experts != self.num_experts:
             raise ValueError(f"Number of experts in routing_weights ({N_experts}) must match module experts ({self.num_experts}).")

        effective_kernel = torch.einsum("bi, eocdwh->bocdwh", routing_weights, self.weight)
        effective_bias = None
        if self.bias is not None:
            effective_bias = torch.einsum("bi, eo->bo", routing_weights, self.bias)

        x_reshaped = x.view(1, B * C_in, D, H, W)
        effective_kernel_reshaped = effective_kernel.view(
            B * self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        effective_bias_reshaped = None
        if effective_bias is not None:
            effective_bias_reshaped = effective_bias.view(B * self.out_channels)

        output = F.conv3d(
            x_reshaped,
            effective_kernel_reshaped,
            effective_bias_reshaped,
            self.stride,
            self.padding,
            self.dilation,
            B * self.groups
        )

        output = output.view(B, self.out_channels, *output.shape[2:])

        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, padding={padding}')
        if self.dilation != _triple(1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is False:
            s += ', bias=False'
        s += ', num_experts={num_experts})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class DynamicConv3D(nn.Module):
    """ Dynamic Conv layer for 3D inputs using CondConv3d
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=2, padding=1, dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        print('+++ Initializing DynamicConv3D with', num_experts, 'experts')

        kernel_size_t = _triple(kernel_size)
        stride_t = _triple(stride)
        padding_t = _triple(padding)
        dilation_t = _triple(dilation)

        self.routing = nn.Linear(in_features, num_experts)

        self.cond_conv = CondConv3d(
            in_channels=in_features, 
            out_channels=out_features,
            kernel_size=kernel_size_t,
            stride=stride_t,
            padding=padding_t,
            dilation=dilation_t,
            groups=groups,
            bias=bias,
            num_experts=num_experts
        )

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool3d(x, (1, 1, 1))
        pooled_inputs = pooled_inputs.flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))

        x = self.cond_conv(x, routing_weights)

        return x
