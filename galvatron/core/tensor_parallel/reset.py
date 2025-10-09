from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
# from torch.nn.init import xavier_uniform_ as init_method
from galvatron.core.tensor_parallel import init_method_normal
from megatron.training import get_args
import torch

def colummn_row_reset_parameters(self):
    args = get_args()
    with get_cuda_rng_tracker().fork():
        init_method = init_method_normal(args.init_method_std)
        init_method(self.weight)
    if hasattr(self, "bias") and self.bias != None:
        with torch.no_grad():
            self.bias.zero_()

def init_reset_parameter():
    ColumnParallelLinear.reset_parameters = colummn_row_reset_parameters
    RowParallelLinear.reset_parameters = colummn_row_reset_parameters
    VocabParallelEmbedding.reset_parameters = colummn_row_reset_parameters
