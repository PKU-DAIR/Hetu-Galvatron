import argparse

def model_args(parser):
    group = parser.add_argument_group(title='Model Arguments')

    group.add_argument(
        "--model_size", type=str, default='bert-base', help="Model size.", choices=['bert-base', 'bert-large']
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a", "--num_attention_heads", type=int, default=12, help="Number of attention heads",
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument("--max_predictions_per_seq", type=int, default=20)
    group.add_argument(
        "--make_vocab_size_divisible_by", type=int, default=128,
        help="Pad the vocab size to be divisible by this value. This is needed for tensor parallelism."
    )
    group.add_argument(
        "--kv_channels", type=int, default=None,
        help="Size of key/value for attention. If None, defaults to hidden_size/num_attention_heads"
    )#some error
    return parser

def layernum_arg_names():
    return ['num_hidden_layers']