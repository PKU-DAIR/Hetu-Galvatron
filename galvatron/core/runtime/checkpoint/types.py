from typing import Protocol
import torch.nn as nn

class LoadModuleFunc(Protocol):
    """Callable contract for the per-module checkpoint loaders
    (``load_llama_module`` / ``load_moe_module`` / ``load_gpt_module``).

    Annotate any ``load_module_func`` parameter with this type so callers get
    parameter-name hints, and the implementations are structurally checked
    against a single source of truth.
    """

    def __call__(
        self,
        load: str,
        tp_groups,
        name: str,
        submodule: nn.Module,
        module: nn.Module,
        distributed_checkpoint: bool,
        ep_groups=None,
    ) -> None: ...