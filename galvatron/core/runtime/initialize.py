from contextlib import contextmanager
import os
import time
import json
import torch
import torch.nn as nn

from galvatron.core.runtime.parallel_state import set_global_variables, set_global_memory_buffer
from galvatron.core.runtime.utils.rerun_state_machine import initialize_rerun_state_machine
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from datetime import timedelta
from galvatron.utils import set_seed


@contextmanager
def init_empty_weights(include_buffers: bool = True):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = True):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def _initialize_distributed(args:GalvatronRuntimeArgs):
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)

        torch.cuda.set_device(args.local_rank)

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)


def initialize_galvatron(args:GalvatronRuntimeArgs):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])

    validate_args(args)
    set_global_variables(args)
    _initialize_distributed(args)
    set_seed(args.train.seed)
    set_global_memory_buffer()
    initialize_rerun_state_machine()

    # Setup MoE aux loss scale value.
    if args.model.num_moe_experts is not None:
        from galvatron.core.runtime.moe.router import MoEAuxLossAutoScaler

        MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    _compile_dependencies()


def _compile_dependencies():

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    start_time = time.time()
    if torch.distributed.get_rank() == 0:
        print("> compiling dataset index builder ...")
        from galvatron.core.runtime.datasets.megatron.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling dataset index builder. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def validate_args(args:GalvatronRuntimeArgs):
    train = args.train
    data = args.data
    ckpt = args.ckpt

    # ---------- data ----------
    assert data.num_dataset_builder_threads > 0, "num_dataset_builder_threads must be > 0"
    if data.data_path is not None and data.split is None:
        legacy_split = "969, 30, 1"
        data.split = legacy_split
        if args.rank == 0:
            print(
                "WARNING: Please specify data.split when using data_path. "
                f'Using legacy default "{legacy_split}"',
                flush=True,
            )

    # ---------- iteration-based vs sample-based  ----------
    if train.train_iters is not None:
        assert train.train_samples is None, "Use either train_iters (iteration-based) or train_samples (sample-based), not both"
        assert train.lr_decay_samples is None, "Expected iteration-based training (no lr_decay_samples)"
        assert (train.lr_warmup_samples or 0) == 0, "Expected iteration-based learning rate warmup (no lr_warmup_samples)"
        assert train.rampup_batch_size is None, "Expected no rampup_batch_size for iteration-based training"
        if train.lr_warmup_fraction is not None:
            assert (train.lr_warmup_iters or 0) == 0, "Specify only one of lr_warmup_fraction and lr_warmup_iters"

    if train.train_samples is not None:
        assert train.train_iters is None, "Use either train_iters or train_samples, not both"
        assert train.lr_decay_iters is None, "Expected sample-based learning rate decay (no lr_decay_iters)"
        assert (train.lr_warmup_iters or 0) == 0, "Expected sample-based learning rate warmup (no lr_warmup_iters)"
        if train.lr_warmup_fraction is not None:
            assert (train.lr_warmup_samples or 0) == 0, "Specify only one of lr_warmup_fraction and lr_warmup_samples"

    # ---------- learning rate and weight decay ----------
    if train.lr is not None and train.min_lr is not None:
        assert train.min_lr <= train.lr, "min_lr must be <= lr"
    if train.weight_decay_incr_style == "constant":
        if train.start_weight_decay is None:
            train.start_weight_decay = train.weight_decay
        if train.end_weight_decay is None:
            train.end_weight_decay = train.weight_decay
    else:
        assert train.start_weight_decay is not None, "start_weight_decay required when weight_decay_incr_style != constant"
        assert train.end_weight_decay is not None, "end_weight_decay required when weight_decay_incr_style != constant"

    # ---------- ckpt ----------
    if ckpt.save is not None:
        assert ckpt.save_interval is not None, "save_interval must be set when save is set"


def _print_args(args:GalvatronRuntimeArgs, title: str = "arguments"):
    """Print Pydantic args as indented JSON. Only rank 0 prints."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    d = args.model_dump()
    s = json.dumps(d, indent=2, default=str)
    print(f"\n=== {title} ===\n{s}\n", flush=True)