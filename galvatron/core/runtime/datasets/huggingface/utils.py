from pathlib import Path
from typing import List, Union

def get_data_files(data_paths: Union[str, List[str]]):
    data_files = []
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    for data_path in data_paths:
        if data_path.startswith("hdfs://"):
            raise NotImplementedError("HDFS dataset path is not supported yet.")
        else:
            data_path = Path(data_path)
            if data_path.is_dir():
                data_files.extend([str(file_path) for file_path in data_path.iterdir() if file_path.is_file()])
            elif data_path.is_file():
                data_files.append(str(data_path))
            else:
                raise FileNotFoundError(f"Dataset {data_path} not exists.")
    if not data_files:
        raise ValueError(f"No data files found for data_paths={data_paths!r}")
    
    file_extension = Path(data_files[0]).suffix.lstrip(".").lower()
    if file_extension == "jsonl":
        file_extension = "json"
    if file_extension not in ["parquet", "json", "csv", "arrow"]:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return data_files, file_extension


def get_text_from_example(example: dict, text_keys: Union[str, List[str]]):
    if isinstance(text_keys, str):
        return example.get(text_keys)
    for key in text_keys:
        if key in example:
            return example[key]
    return None


def tokenize_text(text, tokenizer, eos_token_id) -> List[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        # Prefer tokenizer-native defaults (HF fast tokenizer path),
        # and only fallback to legacy bos/eos kwargs when required.
        try:
            ids = tokenizer.tokenize(text)
        except TypeError:
            ids = tokenizer.tokenize(text, bos=False, eos=False)
        if not isinstance(ids, list):
            ids = list(ids)
    if eos_token_id is not None:
        ids = ids + [eos_token_id]
    return ids


def split_into_chunks(token_ids: List[int], chunk_len: int):
    chunks = []
    for i in range(0, len(token_ids), chunk_len):
        chunk = token_ids[i : i + chunk_len]
        if chunk:
            chunks.append(chunk)
    return chunks