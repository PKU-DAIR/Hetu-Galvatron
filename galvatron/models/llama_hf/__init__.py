from .LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model, llama_model_hp
import megatron
import torch 

def rms_reset_parameters(self):
    with torch.no_grad():
        torch.nn.init.ones_(self.weight)
megatron.legacy.model.rms_norm.RMSNorm.reset_parameters = rms_reset_parameters
