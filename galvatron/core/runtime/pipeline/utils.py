from typing import List, Optional, Union

import torch


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]


def chunk_batch(inputs, chunks, cu_seqlens=None):
    if cu_seqlens is not None:
        return chunk_batch_for_pack(inputs, chunks, cu_seqlens)
    else:
        return chunk_batch_for_padding(inputs, chunks)


def chunk_dict(kwargs, chunks):
    if 'cu_seqlens' in kwargs and kwargs['cu_seqlens'] is not None:
        return chunk_dict_for_pack(kwargs, chunks)
    else:
        return chunk_dict_for_padding(kwargs, chunks)


def chunk_batch_for_padding(inputs, chunks):
    if inputs is None:
        return inputs

    batches = [[] for _ in range(chunks)]
    # Actual number of chunks produced
    num_chunks = -1
    for input in inputs:
        if torch.is_tensor(input):
            # Chunk only tensors.
            tensors = input.chunk(chunks)

            # Validate number of chunks equal across all inputs.
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(
                    f"Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}"
                )
            num_chunks = len(tensors)

            for i, tensor in enumerate(tensors):
                batches[i].append(tensor)
        else:
            # Replicate non-tensors or tensors wrapped with 'NoChunk'.
            for i in range(chunks):
                batches[i].append(input)
            num_chunks = chunks

    # Truncate to actual number of chunks
    batches = batches[:num_chunks]

    return batches


def chunk_dict_for_padding(kwargs, chunks):
    batches = [{} for _ in range(chunks)]
    num_chunks = -1
    for k, v in kwargs.items():
        if torch.is_tensor(v) and not (k.endswith("_mask") and v.shape[0] == 1) and not k.startswith("rotary"):
            tensors = v.chunk(chunks)
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(
                    f"Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}"
                )
            num_chunks = len(tensors)
            for i, tensor in enumerate(tensors):
                batches[i][k] = tensor
        else:
            for i in range(chunks):
                batches[i][k] = v

    if num_chunks >= 0:
        batches = batches[:num_chunks]
    return batches


def chunk_batch_for_pack(inputs, chunks, cu_seqlens):
    if inputs is None:
        return inputs

    # NOTE: the shape here is intentionally awkward — input_ids is wrapped in a single-element
    # list purely to stay compatible with the previous chunk_batch_for_padding contract (which
    # takes a list of positional inputs and returns a list-of-lists). That's the only reason
    # for the extra `[...]` layer; otherwise we'd just pass the tensor directly.
    assert len(inputs) == 1, (
        f"expected a single packed input_ids tensor wrapped in a list (the extra list layer "
        f"is kept only for compatibility with chunk_batch_for_padding's prior contract), "
        f"got len(inputs)={len(inputs)}"
    )
    real_inputs = inputs[0]

    # TODO: revisit this once dyn_bsz is enabled. In pack mode under dyn_bsz the
    # number of samples per micro-batch is no longer fixed, so deriving the chunk
    # boundaries from a uniform `batch_size // chunks` split of cu_seqlens is
    # incorrect — the chunking logic needs to be rewritten to honor the dynamic
    # per-chunk sample counts produced by the dyn_bsz scheduler.
    batch_size = cu_seqlens.numel() - 1
    if batch_size % chunks != 0:
        raise RuntimeError(f"Batch size {batch_size} must be divisible by chunks {chunks}")
    samples_per_chunk = batch_size // chunks

    batches = []
    for i in range(chunks):
        t_start = int(cu_seqlens[i * samples_per_chunk].item())
        t_end = int(cu_seqlens[(i + 1) * samples_per_chunk].item())
        batches.append([real_inputs[:, t_start:t_end]])

    return batches


def chunk_dict_for_pack(kwargs, chunks):
    assert set(kwargs.keys()) == {"labels", "cu_seqlens", "rotary_embedding"}, (
        f"kwargs must contain only labels, cu_seqlens, and rotary_embedding in packing mode, got {list(kwargs.keys())}"
    )

    cu_seqlens = kwargs.get("cu_seqlens")
    assert cu_seqlens is not None, "cu_seqlens must be provided in kwargs for packing mode"

    # TODO: revisit this once dyn_bsz is enabled. In pack mode under dyn_bsz the
    # number of samples per micro-batch is no longer fixed, so deriving the chunk
    # boundaries from a uniform `batch_size // chunks` split of cu_seqlens is
    # incorrect — the chunking logic needs to be rewritten to honor the dynamic
    # per-chunk sample counts produced by the dyn_bsz scheduler.
    batch_size = cu_seqlens.numel() - 1
    if batch_size % chunks != 0:
        raise RuntimeError(f"Batch size {batch_size} must be divisible by chunks {chunks}")

    batches = []
    samples_per_chunk = batch_size // chunks
    for i in range(chunks):
        t_start = int(cu_seqlens[i * samples_per_chunk].item())
        t_end = int(cu_seqlens[(i + 1) * samples_per_chunk].item())
        batch = {
            "labels": kwargs["labels"][:, t_start:t_end],
            "cu_seqlens": cu_seqlens[i * samples_per_chunk:(i + 1) * samples_per_chunk + 1] - t_start,
            "rotary_embedding": kwargs["rotary_embedding"], # rotary_embedding is the same for all chunks since it's based on max_seq_len, not actual seq_len. We can optimize this later if needed by chunking the rotary_embedding as well, but it should be fine since it's usually small compared to input_ids and labels.
        }
        batches.append(batch)
    
    return batches

