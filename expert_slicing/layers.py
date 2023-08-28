import torch
from torch import nn
from utils import *
from initialize import get_tensor_model_parallel_world_size

# Column Parallel Linear layer
class ColumnParallelLinear(torch.nn.Module):
    # Initialize function
    def __init__(self, input_size, output_size, bias=True, gather_output=True, skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()
        # Get input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        # world_size = int(os.environ['WORLD_SIZE'])
        # world_size = int(os.getenv('TP_SIZE'))
        world_size = get_tensor_model_parallel_world_size()
        # self.output_size_per_partition = output_size//world_size
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        # Initialize the original weight of the model
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition,
            self.input_size,
            device=torch.cuda.current_device(),
            dtype=torch.float
        ))
        # Initialize the weight
        nn.init.xavier_uniform_(self.weight)
        # Add the bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device(), 
                dtype=torch.float
            ))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = nn.functional.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # output = OutputAdapter.apply(output_parallel)
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False, 
                 skip_bias_add=False
                 ):
        super(RowParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        # world_size = int(os.environ['WORLD_SIZE'])
        # world_size = int(os.getenv('TP_SIZE'))
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        print(self.output_size," ", self.input_size_per_partition)
        self.weight = nn.Parameter(torch.empty(
                self.output_size, 
                self.input_size_per_partition,
                device=torch.cuda.current_device(), 
                dtype=torch.float
        ))
        # Initialize the weight
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(
                    self.output_size, 
                    device=torch.cuda.current_device(),
                    dtype=torch.float
            ))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        output_parallel = nn.functional.linear(input_parallel, self.weight)
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

bias_gelu_impl = GeLUFunction.apply

class ParallelMLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, hidden_size, ffn_hidden_size, bias_gelu_fusion=True, openai_gelu=False, onnx_safe=False):
        super(ParallelMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            gather_output=False,
            skip_bias_add=True)

        self.bias_gelu_fusion = bias_gelu_fusion
        self.activation_func = nn.functional.gelu
        if openai_gelu:
            self.activation_func = openai_gelu
        elif onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            input_is_parallel=True,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias