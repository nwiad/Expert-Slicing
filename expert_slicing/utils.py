import torch

_TENSOR_MODEL_PARALLEL_GROUP=None
_MPU_TENSOR_MODEL_PARALLEL_RANK=None


'''
下面两个函数都是用来辅助切割的,不需要修改
'''
# 保证能整除
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

# 整除函数
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


# 按照最后一个dimension进行切割的函数
def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if 2==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=torch.distributed.GroupMember.WORLD)

    return input_

def _split(input_, cut_size=2):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = cut_size
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank()
    output = input_list[rank].contiguous()

    return output
'''
----------Gather function----------
'''

def _gather(input_, cut_size=2):
    """Gather tensors and concatinate along the last dimension."""
    world_size = cut_size
    if world_size==1:
        return input_
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return input_
    @staticmethod
    def forward(ctx, input_):
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""
    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        #print("phase 3 backward executed")
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        #print("phase 1 forward executed")
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        #print("phase 1 backward executed")
        return _gather(grad_output)
class OutputAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_)
        output = torch.cat(tensor_list, dim=-1)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        tensor_list = torch.split(
            grad_output, grad_output.size()[-1]//world_size, dim=-1
        )
        return tensor_list[rank].contiguous()

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)