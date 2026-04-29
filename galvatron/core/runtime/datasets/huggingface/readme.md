# HuggingFace Data Pipeline

## 1. Overview
This pipeline enables Galvatron to leverage the HuggingFace `datasets` ecosystem while maintaining compatibility with high-performance execution paths. It bridges the flexibility of HuggingFace (streaming, diverse file formats) with the high-throughput requirements of distributed LLM training, ensuring that data loading never becomes a bottleneck for the GPU.

---

## 2. Configuration Parameters
These settings are defined under the `data:` section of your configuration and are active only when `data_source` is set to `hf`.

### Core Pipeline Settings
- **data_source**: Set to `hf` to enable this specific pipeline.
- **train_data_path**: A list of filesystem paths. Galvatron automatically expands directories and identifies supported formats (Parquet, JSON, CSV, Arrow).
- **valid_data_path / test_data_path**: Optional paths for validation and testing; iterators are only built if these are provided.

### HuggingFace Specific Knobs
- **hf_data_mode**: Selects the data lifecycle: `prefetch` (asynchronous), `iterable` (streaming), or `mapping` (materialized).
- **hf_text_keys**: Column name(s) for raw text. It supports fallback logic (first matching key wins).
- **hf_collator_mode**: `padding` for fixed-length batches or `packing` for variable-length sequence concatenation.
- **hf_shuffle_buffer_size**: Window size for streaming shuffles. Crucial for randomness in `prefetch` and `iterable` modes.
- **hf_num_workers**: Controls parallelism for tokenization and batch production.
- **hf_prefetch_factor**: Defines the depth of the shared-memory ring buffer in `prefetch` mode.

---

## 3. Data Processing Strategies

### The Three Loading Modes
1. **prefetch (Default)**: Optimized for maximum throughput. It offloads "Read + Tokenize + Batch" logic to a dedicated sub-process. Batches are pushed into a shared-memory buffer, allowing the training process to fetch ready tensors instantly.
2. **iterable**: A standard streaming approach. It tokenizes and yields data chunks on-the-fly, suitable for environments where multiprocess shared memory is restricted or for simpler workflows.
3. **mapping**: Best for datasets that fit on disk. It materializes the data, performs tokenization once via `dataset.map`, and supports random access through a standard `DistributedSampler`.

### Collation & Chunking
- **Fixed-length Chunks**: Regardless of the mode, the pipeline abstracts raw text into fixed-length token chunks (based on `seq_length`).
- **Padding vs. Packing**: The collator prepares these chunks into the final micro-batch layout, either by padding sequences or packing multiple samples into a single block to minimize computation waste.

---

## 4. System Design: The Architecture of Speed
The system is divided into two primary layers that work in tandem to ensure data flows efficiently from storage to the GPU.

### Data Layer: `dataset.py`
This layer is responsible for **"where the data comes from and how it is sharded."** It handles the abstraction from raw files to fixed-length tokenized chunks.

- **Materialized Path (Mapping)**: Uses non-streaming `load_dataset`. Data is tokenized via batched maps and split into chunks upfront, creating a materialized dataset that supports `__getitem__`.
- **Streaming Path (Iterable)**: Implements real-time tokenization and yielding. 
- **Distributed Awareness**: To avoid the "empty shard" problem common in streaming libraries, Galvatron uses custom sharding logic:
    - For **single-file** sources: A round-robin approach (`idx % dp_world_size == dp_rank`) ensures every GPU gets a balanced share.
    - For **multi-file** sources: The stream is split into a fixed number of sub-shards, which are then interleaved (`interleave_datasets`) based on the rank's index.

### Performance Layer: `prefetch_strategy.py`
This layer handles **"asynchronous batch production and GPU alignment"** when `hf_data_mode=prefetch` is active.

- **The Forked Producer**: To hide I/O and CPU-heavy tokenization latency, the pipeline forks a producer process. This process masks its own GPUs (`CUDA_VISIBLE_DEVICES=""`) to prevent resource conflicts.
- **Shared Memory Ring Buffer**: 
    - The producer fills "slots" in a shared-memory buffer with tensors that are already formatted for the GPU (supporting both padding and packing layouts).
    - It uses a `SimpleQueue` to pass metadata (slot indices) to the parent process.
- **Flow Control**: The parent process reads from the buffer and uses semaphores to manage the number of "in-flight" batches, creating a circular prefetch depth defined by `prefetch_factor`.
- **Safe Initialization**: The system ensures that the fork happens **after** the global initialization of Megatron/Galvatron. This allows the child process to inherit the initialized tokenizer and configuration state while maintaining a clean execution environment.

### Summary of Synergy
The architecture ensures a clean separation of concerns: `dataset.py` solves the **data distribution and sharding** across multiple GPUs, while `prefetch_strategy.py` solves the **latency overlap** problem by ensuring that while the GPU is busy calculating one step, the next batch is already sitting in shared memory, tokenized and ready to go.