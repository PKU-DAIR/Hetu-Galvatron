"""Module registry and architecture metadata.

Central registry that maps declarative module type names (e.g. ``"decoder"``)
to their concrete ``nn.Module`` classes, plus ``ArchModelInfo`` which
auto-derives ModelInfo from an architecture list.
"""

from typing import Dict, List, Type
from dataclasses import dataclass

import torch.nn as nn

from galvatron.core.runtime.hybrid_parallel_config import mixed_precision_dtype
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs

from .modules import (
    GalvatronEmbedding,
    GalvatronDecoderLayer,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
    GalvatronMoEDecoderLayer,
)


# =========================================================================
# Type constants
# =========================================================================

_LAYER_MODULE_TYPES = {"decoder", "moe_decoder"}
"""Module types that count as repeating "layers" for parallel config."""

_MODULE_TYPE_SUFFIX = {
    "embedding": "embed",
    "decoder": "dec",
    "moe_decoder": "moe_dec",
    "prenorm": "norm",
    "lm_head": "cls",
}
"""Module type → suffix used by ``hp_config_whole_model``."""

MODULE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "embedding": GalvatronEmbedding,
    "decoder": GalvatronDecoderLayer,
    "moe_decoder": GalvatronMoEDecoderLayer,
    "prenorm": GalvatronFinalNorm,
    "lm_head": GalvatronCausalLMHead,
}
"""Module type → concrete class."""


# =========================================================================
# Helpers
# =========================================================================

def arch_to_module_types(arch_list: List[str]) -> List[str]:
    """Convert an architecture list to the ``module_types`` format expected by Galvatron."""
    return [_MODULE_TYPE_SUFFIX.get(t, t) for t in arch_list]


# =========================================================================
# ModelInfo
# =========================================================================
class ModelInfo:
    def __init__(self):
        return

    def set_layernums(self, info):
        self.layernum_list = info

    def set_shapes(self, info):
        self.layer_shapes_list = info

    def set_dtypes(self, info):
        self.layer_dtypes_list = info

    def set_module_types(self, info):
        self.layer_module_types = info

    def layernums(self):
        return self.layernum_list

    def shapes(self):
        return self.layer_shapes_list

    def dtypes(self):
        return self.layer_dtypes_list

    def module_types(self):
        return self.layer_module_types


# =========================================================================
# Auto-derived ModelInfo
# =========================================================================

class ArchModelInfo(ModelInfo):
    """``ModelInfo`` automatically derived from *arch_list* + *args*."""

    def __init__(self, arch_list: List[str], args:GalvatronRuntimeArgs):
        super().__init__()
        m = args.model
        if m.model_type in ["gpt", "llama", "qwen", "mistral", "mixtral"]:
            num_layers = m.num_layers
            seq_len = args.train.seq_length
            hidden_size = m.hidden_size
            mp_dtype = mixed_precision_dtype(args.parallel.mixed_precision)

            if m.shape_order == "SBH":
                layer_shapes = [[[seq_len, -1, hidden_size]]]
            else:
                layer_shapes = [[[-1, seq_len, hidden_size]]]

            module_types = arch_to_module_types(arch_list) # TODO: Check if it is necessary

            self.set_layernums([num_layers])
            self.set_shapes(layer_shapes)
            self.set_dtypes([[mp_dtype]])
            self.set_module_types(module_types)
        else:
            assert False, "Unknown model type: " + m.model_type


# =========================================================================
# BlockNames
# =========================================================================
@dataclass
class BlockNames:
    wrap_block_name: List[nn.Module]
    wrap_checkpoint_block_name: List[nn.Module]
    wrap_other_block_name: List[nn.Module]
    all_block_name: List[nn.Module]
