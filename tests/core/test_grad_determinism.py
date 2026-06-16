"""
FSDP1 vs FSDP2 gradient determinism test — multi-micro-batch + no_sync + manual reduce.

Compares gradients at TWO points:
  (A) After no_sync accumulation, BEFORE manual gradient reduction
  (B) After manual gradient reduction (mimics fsdp_reduce_gradients)

Usage (8 GPUs):
    torchrun --nproc_per_node=8 tests/core/test_grad_determinism.py

Flags:
    --no-optimizer       Disable optimizer (isolate backward pass)
    --seed N             Random seed (default 42)
    --train-iters N      Number of training iterations (default 10)
    --micro-batches N    Number of micro-batches per iteration (default 4)
    --num-experiments N  Number of full experiments (default 2)
"""

import argparse
import hashlib
import os

# Determinism: CUBLAS_WORKSPACE_CONFIG must be set BEFORE any CUDA/cuBLAS context
# is created, otherwise torch.use_deterministic_algorithms(True) will error on
# deterministic cuBLAS matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._common_utils import TrainingState, _FSDPState
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState as FSDPv2State
from torch.distributed.fsdp._runtime_utils import _unshard, _post_backward_final_callback
from torch.distributed.tensor import DTensor

# Handle torch version differences for FSDP1 internals
try:
    from torch.distributed.fsdp._flat_param import (
        FlatParamHandle, HandleTrainingState,
    )
except ImportError:
    from torch.distributed.fsdp.flat_param import (
        FlatParamHandle, HandleTrainingState,
    )

from galvatron.core.runtime.pipeline.sp_grad_reduce import _post_backward_hook_sp as _post_backward_hook


# ---------------------------------------------------------------------------
# Tiny transformer model
# ---------------------------------------------------------------------------

class SimpleSelfAttention(nn.Module):
    """Manual self-attention — avoids nn.MultiheadAttention view bugs with FSDP1."""

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(attn_out)


class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = SimpleSelfAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        mlp_out = self.linear2(torch.nn.functional.silu(self.linear1(x)))
        x = self.norm2(x + mlp_out)
        return x


class TinyModel(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=256, intermediate_size=1024,
                 num_layers=2, num_heads=8, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(*[
            SimpleDecoderLayer(hidden_size, intermediate_size, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x = self.layers(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Gradient checksum
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_local_grads(model):
    """
    Return this rank's local grad tensors in a consistent FSDP-unit order,
    handling FSDP1/FSDP2 and BOTH pre-reduce (no_sync accumulation) and post-reduce.

    FSDP1: flat_param.grad (pre: full unsharded; post: may live in _saved_grad_shard).
    FSDP2: param.grad is None during no_sync — the accumulated grad lives in
           unsharded_accumulated_grad (pre); after reduce it's sharded_param.grad (post).
           Reading param.grad directly (as the old code did) yielded None → zero hash.
    """
    grads = []
    for m in model.modules():
        if isinstance(m, FSDP1):
            if hasattr(m, "_handles"):
                handles = m._handles
            elif getattr(m, "_handle", None) is not None:
                handles = [m._handle]
            else:
                continue
            for handle in handles:
                fp = handle.flat_param
                g = fp.grad
                if g is None and getattr(fp, "_saved_grad_shard", None) is not None:
                    g = fp._saved_grad_shard
                if g is not None:
                    grads.append(g.detach())
        elif isinstance(m, FSDPModule):
            try:
                state = m._get_fsdp_state()
            except Exception:
                continue
            pg = state._fsdp_param_group
            if pg is None:
                continue
            for fsdp_param in pg.fsdp_params:
                acc = fsdp_param.unsharded_accumulated_grad
                if acc is not None:                                   # pre-reduce (no_sync)
                    g = acc.data
                elif fsdp_param.unsharded_param is not None and fsdp_param.unsharded_param.grad is not None:
                    g = fsdp_param.unsharded_param.grad.data           # pre-reduce (synced)
                elif fsdp_param.sharded_param is not None and fsdp_param.sharded_param.grad is not None:
                    g = fsdp_param.sharded_param.grad                  # post-reduce
                else:
                    continue
                if isinstance(g, DTensor):
                    g = g.to_local()
                grads.append(g.detach())
    return grads


@torch.no_grad()
def grad_checksum(model, rank):
    """Compute global grad_mean, grad_l2, and bitwise hash (all-reduced)."""
    grad_sum_local = 0.0
    grad_sq_local = 0.0
    numel_local = 0
    hasher = hashlib.sha256()

    for grad in _collect_local_grads(model):
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        grad_sum_local += grad.sum().item()
        grad_sq_local += grad.pow(2).sum().item()
        numel_local += grad.numel()
        grad_fp32 = grad.float().contiguous().cpu()
        hasher.update(grad_fp32.numpy().tobytes())

    def _ar(t):
        tc = torch.tensor([t], dtype=torch.float64, device="cuda")
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        return tc.item()

    grad_sum = _ar(grad_sum_local)
    grad_sq = _ar(grad_sq_local)
    total_numel = _ar(numel_local)

    grad_mean = grad_sum / max(total_numel, 1)
    grad_l2 = grad_sq / max(total_numel, 1)

    hbytes = hasher.digest()
    hint = int.from_bytes(hbytes[:8], 'little', signed=True)
    hc = torch.tensor([hint], dtype=torch.int64, device="cuda")
    all_h = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(dist.get_world_size())]
    dist.all_gather(all_h, hc)
    global_hash = 0
    for h in all_h:
        global_hash ^= h.item()

    return grad_mean, grad_l2, global_hash


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------

def make_fsdp1_model(base_model, mp_dtype, device):
    """Wrap each decoder layer + root (matching FSDP2's wrapping structure)."""
    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,              # bf16 reduce → matches "reduce_in_fp32: false"
        buffer_dtype=mp_dtype,
        cast_forward_inputs=False,
        cast_root_forward_inputs=False,
    )
    # Wrap each decoder layer individually (like FSDP2's fully_shard per layer)
    for i, layer in enumerate(base_model.layers):
        base_model.layers[i] = FSDP1(
            layer,
            process_group=None,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mp_policy,
            device_id=device,
        )
    # Wrap the root
    return FSDP1(
        base_model,
        process_group=None,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mp_policy,
        device_id=device,
    )


def make_fsdp2_model(base_model, mp_dtype, device):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,              # bf16 reduce → matches "reduce_in_fp32: false"
        output_dtype=mp_dtype,
        cast_forward_inputs=False,
    )
    mesh = DeviceMesh.from_group(dist.group.WORLD, device_type="cuda")
    for layer in base_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
    return fully_shard(base_model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)


# ---------------------------------------------------------------------------
# Manual gradient reduction (mimicking Galvatron's fsdp_reduce_gradients)
# ---------------------------------------------------------------------------

def manual_reduce_gradients_fsdp1(model):
    """FSDP1: trigger reduce-scatter on accumulated unsharded gradients."""
    for m in model.modules():
        if not isinstance(m, FSDP1):
            continue
        m.training_state = TrainingState.FORWARD_BACKWARD
        if hasattr(m, "_handles"):
            for handle in m._handles:
                handle._training_state = HandleTrainingState.BACKWARD_PRE
                _unshard(m, m._handles, m._streams["unshard"], m._streams["pre_unshard"])
                _post_backward_hook(m, handle, None)
        elif m._handle is not None:
            m._handle._training_state = HandleTrainingState.BACKWARD_PRE
            _unshard(m, m._handle, m._unshard_stream, m._pre_unshard_stream)
            _post_backward_hook(m, m._handle, None)
    for m in model.modules():
        if isinstance(m, FSDP1) and m._is_root:
            _post_backward_final_callback(m, m)


def manual_reduce_gradients_fsdp2(model):
    """FSDP2: trigger post_backward reduce-scatter."""
    root_states = []
    for m in model.modules():
        if not isinstance(m, FSDPModule):
            continue
        state = m._get_fsdp_state()
        param_group = state._fsdp_param_group
        if param_group is not None:
            param_group.post_backward()
        state._state_ctx.is_last_backward = True
        if state._is_root:
            root_states.append(state)
    for state in root_states:
        state._root_post_backward_final_callback()


# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------

def get_deterministic_data(batch_size, seq_len, vocab_size, seed, num_iters, micro_batches):
    """Generate same data for every run. Returns list of (micro_batch_list, labels_list)."""
    g = torch.Generator()
    g.manual_seed(seed)
    all_steps = []
    for _ in range(num_iters):
        mbs = []
        for _ in range(micro_batches):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
            mbs.append(ids)
        all_steps.append(mbs)
    return all_steps


# ---------------------------------------------------------------------------
# Per-layer gradient debug (inline, similar to grad_debug.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _print_layer_grads(model, label, rank, is_post=False):
    """
    Print per-FSDP-unit flat_sum and numel.
    PRE-reduce (is_post=False): rank0 only, no collectives.
    POST-reduce (is_post=True): all ranks must participate (full_tensor is collective).
    """
    if not is_post and rank != 0:
        return

    if rank == 0:
        print(f"\n  [{label}]")
    unit_idx = 0
    for m in model.modules():
        if isinstance(m, FSDP1):
            if hasattr(m, "_handles"):
                handles = m._handles
            elif m._handle is not None:
                handles = [m._handle]
            else:
                continue
            for handle in handles:
                fp = handle.flat_param
                grad = fp.grad
                if grad is None and hasattr(fp, "_saved_grad_shard") and fp._saved_grad_shard is not None:
                    grad = fp._saved_grad_shard
                if grad is None:
                    continue
                g = grad.detach()
                if rank == 0:
                    print(f"    unit {unit_idx}: numel={g.numel():>10d}  sum={g.sum().item():+18.8e}")
                unit_idx += 1

        elif isinstance(m, FSDPModule):
            try:
                state = m._get_fsdp_state()
            except Exception:
                continue
            pg = state._fsdp_param_group
            if pg is None:
                continue
            grads = []
            total_n = 0
            for fp in pg.fsdp_params:
                g = None
                # PRE-reduce: accumulated grad is in unsharded_accumulated_grad
                acc = fp.unsharded_accumulated_grad
                if acc is not None:
                    g = acc.data.detach()
                # PRE-reduce fallback: unsharded_param.grad
                elif fp.unsharded_param is not None and fp.unsharded_param.grad is not None:
                    g = fp.unsharded_param.grad.data.detach()
                # POST-reduce: gradient is in sharded_param.grad (DTensor → full_tensor is collective!)
                elif fp.sharded_param is not None and fp.sharded_param.grad is not None:
                    grad_dt = fp.sharded_param.grad
                    if isinstance(grad_dt, DTensor):
                        g = grad_dt.full_tensor()  # collective all_gather
                    else:
                        g = grad_dt.detach()
                if g is None:
                    continue
                grads.append(g.reshape(-1))
                total_n += g.numel()
            if len(grads) == 0:
                continue
            cat = torch.cat(grads, dim=0)
            if rank == 0:
                print(f"    unit {unit_idx}: numel={total_n:>10d}  sum={cat.sum().item():+18.8e}")
            unit_idx += 1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_fsdp(model, all_steps, rank, device, use_optimizer, model_label, micro_batches,
                   reduce_func):
    results = {
        "loss": [], "grad_mean_pre": [], "grad_l2_pre": [], "grad_hash_pre": [],
        "grad_mean_post": [], "grad_l2_post": [], "grad_hash_post": [], "grad_norm": [],
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if use_optimizer else None
    is_fsdp2 = isinstance(model, FSDPModule)

    for it, micro_data in enumerate(all_steps):
        total_loss = 0.0

        # --- Phase 1: no_sync micro-batch accumulation ---
        if is_fsdp2:
            model.set_requires_gradient_sync(False, recurse=True)
            for mb_ids in micro_data:
                mb_ids = mb_ids.to(device)
                labels = mb_ids[:, 1:]
                input_ids = mb_ids[:, :-1]
                loss = model(input_ids, labels=labels)
                loss.backward()
                total_loss += loss.item()
            model.set_requires_gradient_sync(True, recurse=True)
        else:
            with model.no_sync():
                for mb_ids in micro_data:
                    mb_ids = mb_ids.to(device)
                    labels = mb_ids[:, 1:]
                    input_ids = mb_ids[:, :-1]
                    loss = model(input_ids, labels=labels)
                    loss.backward()
                    total_loss += loss.item()

        avg_loss = total_loss / micro_batches

        # --- Per-layer PRE-reduce debug (rank0 only, no collectives) ---
        _print_layer_grads(model, f"{model_label} PRE-reduce (rank0 local)", rank, is_post=False)

        # --- Checksum (A): after accumulation, BEFORE manual reduction ---
        gm_pre, gl2_pre, gh_pre = grad_checksum(model, rank)

        # --- Phase 2: manual gradient reduction ---
        reduce_func(model)
        dist.barrier()

        # --- Per-layer POST-reduce debug (all ranks participate for full_tensor) ---
        _print_layer_grads(model, f"{model_label} POST-reduce", rank, is_post=True)

        # --- Checksum (B): AFTER manual reduction ---
        gm_post, gl2_post, gh_post = grad_checksum(model, rank)

        # --- Grad norm ---
        gn_local = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if isinstance(g, DTensor):
                    g = g.to_local()
                gn_local += g.pow(2).sum().item()
        gn_t = torch.tensor([gn_local], device="cuda")
        dist.all_reduce(gn_t, op=dist.ReduceOp.SUM)
        gn_val = gn_t.item() ** 0.5

        results["loss"].append(avg_loss)
        results["grad_mean_pre"].append(gm_pre)
        results["grad_l2_pre"].append(gl2_pre)
        results["grad_hash_pre"].append(gh_pre)
        results["grad_mean_post"].append(gm_post)
        results["grad_l2_post"].append(gl2_post)
        results["grad_hash_post"].append(gh_post)
        results["grad_norm"].append(gn_val)

        if rank == 0:
            print(f"  [{model_label}] it={it:3d} | loss={avg_loss:.8f} | "
                  f"PRE:  mean={gm_pre:+.6e} l2={gl2_pre:.6e} hash={gh_pre:#018x}")
            print(f"  {' ':>13s} | POST: mean={gm_post:+.6e} l2={gl2_post:.6e} hash={gh_post:#018x} "
                  f"gn={gn_val:.4f}")

        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()

        dist.barrier()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--num-experiments", type=int, default=2)
    args = parser.parse_args()

    # ── Determinism: make SDPA backward & cuBLAS reproducible so FSDP1 vs FSDP2
    #    PRE-reduce grads become bitwise-comparable (mirrors flash-attn
    #    deterministic=True in the real runtime). ──
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Force SDPA to the math backend: flash / mem-efficient backward use atomicAdd
    # and are non-deterministic; math is a pure composite (matmul+softmax) and is
    # deterministic given deterministic cuBLAS.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== FSDP1 vs FSDP2 : multi-micro-batch + no_sync + manual reduce ===")
        print(f"  World: {dist.get_world_size()}  Micro-batches: {args.micro_batches}")
        print(f"  Iters: {args.train_iters}  Optimizer: {not args.no_optimizer}")
        print()

    hp = dict(vocab_size=32000, hidden_size=4096, intermediate_size=11008,
              num_layers=4, num_heads=32, seq_len=1024, batch_size=2, mp_dtype=torch.bfloat16)

    all_steps = get_deterministic_data(hp["batch_size"], hp["seq_len"], hp["vocab_size"],
                                       args.seed, args.train_iters, args.micro_batches)

    for exp_idx in range(args.num_experiments):
        if rank == 0:
            print(f"\n{'='*80}\nEXPERIMENT {exp_idx+1}/{args.num_experiments}\n{'='*80}")

        # Save base weights
        torch.manual_seed(args.seed + exp_idx * 1000)
        base = TinyModel(**{k: v for k, v in hp.items()
                            if k in ("vocab_size", "hidden_size", "intermediate_size",
                                     "num_layers", "num_heads")})
        ckpt_path = f"/tmp/fsdp_mb_test_r{rank}_e{exp_idx}.pt"
        torch.save(base.state_dict(), ckpt_path)
        dist.barrier()

        # --- FSDP1 ---
        if rank == 0:
            print("\n--- FSDP1 ---")
        torch.manual_seed(args.seed + exp_idx * 1000)
        m1 = TinyModel(**{k: v for k, v in hp.items()
                          if k in ("vocab_size", "hidden_size", "intermediate_size",
                                   "num_layers", "num_heads")})
        m1.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        m1 = m1.to(device)
        m1 = make_fsdp1_model(m1, hp["mp_dtype"], device)
        dist.barrier()
        r1 = train_one_fsdp(m1, all_steps, rank, device, not args.no_optimizer,
                            "FSDP1", args.micro_batches, manual_reduce_gradients_fsdp1)

        # --- FSDP2 ---
        if rank == 0:
            print("\n--- FSDP2 ---")
        torch.manual_seed(args.seed + exp_idx * 1000)
        m2 = TinyModel(**{k: v for k, v in hp.items()
                          if k in ("vocab_size", "hidden_size", "intermediate_size",
                                   "num_layers", "num_heads")})
        m2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        m2 = m2.to(device)
        m2 = make_fsdp2_model(m2, hp["mp_dtype"], device)
        dist.barrier()
        r2 = train_one_fsdp(m2, all_steps, rank, device, not args.no_optimizer,
                            "FSDP2", args.micro_batches, manual_reduce_gradients_fsdp2)

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        # --- Compare ---
        if rank == 0:
            hdr = f"{'It':>3s} | {'ΔLoss':>12s} | {'PRE Δmean':>13s} | {'PRE Δl2':>13s} | {'PRE H?':>6s} | {'POST Δmean':>13s} | {'POST Δl2':>13s} | {'POST H?':>6s}"
            print(f"\n{'─'*120}\nCOMPARISON {exp_idx+1}\n{'─'*120}\n{hdr}\n{'─'*120}")
            for it in range(args.train_iters):
                dl = abs(r1["loss"][it] - r2["loss"][it])
                d_pre_m = abs(r1["grad_mean_pre"][it] - r2["grad_mean_pre"][it])
                d_pre_l2 = abs(r1["grad_l2_pre"][it] - r2["grad_l2_pre"][it])
                h_pre = "✓" if r1["grad_hash_pre"][it] == r2["grad_hash_pre"][it] else "✗"
                d_post_m = abs(r1["grad_mean_post"][it] - r2["grad_mean_post"][it])
                d_post_l2 = abs(r1["grad_l2_post"][it] - r2["grad_l2_post"][it])
                h_post = "✓" if r1["grad_hash_post"][it] == r2["grad_hash_post"][it] else "✗"
                print(f"{it:3d} | {dl:12.4e} | {d_pre_m:13.4e} | {d_pre_l2:13.4e} | {h_pre:>6s} | "
                      f"{d_post_m:13.4e} | {d_post_l2:13.4e} | {h_post:>6s}")

            # Summary
            losses_ok = sum(1 for it in range(args.train_iters)
                            if abs(r1["loss"][it] - r2["loss"][it]) < 1e-10)
            pre_match = sum(1 for it in range(args.train_iters)
                            if r1["grad_hash_pre"][it] == r2["grad_hash_pre"][it])
            post_match = sum(1 for it in range(args.train_iters)
                             if r1["grad_hash_post"][it] == r2["grad_hash_post"][it])
            print(f"\n  Loss identical: {losses_ok}/{args.train_iters}")
            print(f"  PRE-reduce  hash matches: {pre_match}/{args.train_iters}")
            print(f"  POST-reduce hash matches: {post_match}/{args.train_iters}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
