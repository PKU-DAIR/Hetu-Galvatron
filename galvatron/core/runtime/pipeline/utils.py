from typing import List, Optional, Union

import torch


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]


def chunk_batch(inputs, chunks, kwargs=None):
    # dyn_bsz path: chunk boundaries are explicit (non-uniform sample counts per chunk).
    if kwargs is None:
        return chunk_batch_for_padding(inputs, chunks)
    if kwargs.get("cu_seqlens_chunks") is not None:
        assert kwargs.get("cu_seqlens") is not None, "cu_seqlens_chunks requires cu_seqlens"
        return chunk_batch_for_dyn_bsz(inputs, chunks, kwargs["cu_seqlens"], kwargs["cu_seqlens_chunks"])
    if kwargs.get("cu_seqlens") is not None:
        return chunk_batch_for_pack(inputs, chunks, kwargs["cu_seqlens"])
    return chunk_batch_for_padding(inputs, chunks)


def chunk_dict(kwargs, chunks):
    # dyn_bsz path: presence of cu_seqlens_chunks signals non-uniform per-chunk sample counts.
    if kwargs.get("cu_seqlens_chunks") is not None:
        return chunk_dict_for_dyn_bsz(kwargs, chunks)
    if 'cu_seqlens' in kwargs and kwargs['cu_seqlens'] is not None:
        return chunk_dict_for_pack(kwargs, chunks)
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


def chunk_batch_for_dyn_bsz(inputs, chunks, cu_seqlens, cu_seqlens_chunks):
    """
    Pack-mode chunking for dyn_bsz: each micro-batch has its own (possibly
    unequal) number of inner samples, given by ``cu_seqlens_chunks`` — an
    array of indices into ``cu_seqlens`` marking micro-batch boundaries
    (shape ``(chunks + 1,)``).
    """
    if inputs is None:
        return inputs

    assert len(inputs) == 1, (
        f"expected a single packed input_ids tensor wrapped in a list, got len(inputs)={len(inputs)}"
    )
    real_inputs = inputs[0]

    assert cu_seqlens_chunks.numel() == chunks + 1, (
        f"cu_seqlens_chunks length {cu_seqlens_chunks.numel()} != chunks+1 ({chunks + 1})"
    )

    batches = []
    for i in range(chunks):
        s_start = int(cu_seqlens_chunks[i].item())
        s_end = int(cu_seqlens_chunks[i + 1].item())
        t_start = int(cu_seqlens[s_start].item())
        t_end = int(cu_seqlens[s_end].item())
        batches.append([real_inputs[:, t_start:t_end]])

    return batches


def chunk_dict_for_dyn_bsz(kwargs, chunks):
    """
    Pack-mode kwargs chunking for dyn_bsz. Mirrors ``chunk_dict_for_pack`` but
    uses ``cu_seqlens_chunks`` to determine non-uniform per-micro-batch sample
    counts. ``cu_seqlens_chunks`` itself is consumed here (not forwarded into
    each micro-batch's kwargs, since the per-micro ``cu_seqlens`` is already
    local).
    """
    expected_keys = {"labels", "cu_seqlens", "cu_seqlens_chunks", "rotary_embedding"}
    assert set(kwargs.keys()) == expected_keys, (
        f"kwargs must contain {expected_keys} in dyn_bsz pack mode, got {list(kwargs.keys())}"
    )

    cu_seqlens = kwargs["cu_seqlens"]
    cu_seqlens_chunks = kwargs["cu_seqlens_chunks"]
    assert cu_seqlens is not None and cu_seqlens_chunks is not None, (
        "cu_seqlens and cu_seqlens_chunks must be provided in dyn_bsz pack mode"
    )
    assert cu_seqlens_chunks.numel() == chunks + 1, (
        f"cu_seqlens_chunks length {cu_seqlens_chunks.numel()} != chunks+1 ({chunks + 1})"
    )

    batches = []
    for i in range(chunks):
        s_start = int(cu_seqlens_chunks[i].item())
        s_end = int(cu_seqlens_chunks[i + 1].item())
        t_start = int(cu_seqlens[s_start].item())
        t_end = int(cu_seqlens[s_end].item())
        batch = {
            "labels": kwargs["labels"][:, t_start:t_end],
            "cu_seqlens": cu_seqlens[s_start:s_end + 1] - t_start,
            "rotary_embedding": kwargs["rotary_embedding"],
        }
        batches.append(batch)

    return batches

