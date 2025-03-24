import argparse

def model_args(parser):
    group = parser.add_argument_group(title='ViT Model Arguments')

    group.add_argument(
        "--model_size", type=str, default='vit-base', 
        help="Model size.", 
        choices=['vit-base', 'vit-large', 'vit-huge', 'vit-xhuge']
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, 
        help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, 
        help="Number of hidden layers in the Transformer encoder"
    )
    group.add_argument(
        "--num_attention_heads", type=int, default=12, 
        help="Number of attention heads",
    )
    group.add_argument(
        "--intermediate_size", type=int, default=3072, 
        help="Dimensionality of the feed-forward layer"
    )
    group.add_argument(
        "--hidden_dropout_prob", type=float, default=0.0,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler"
    )
    group.add_argument(
        "--attention_probs_dropout_prob", type=float, default=0.0,
        help="The dropout ratio for the attention probabilities"
    )
    group.add_argument(
        "--layer_norm_eps", type=float, default=1e-12,
        help="The epsilon used by the layer normalization layers"
    )
    group.add_argument(
        "--image_size", type=int, default=224, 
        help="The size (resolution) of each image"
    )
    group.add_argument(
        "--patch_size", type=int, default=16, 
        help="The size (resolution) of each patch"
    )
    group.add_argument(
        "--num_channels", type=int, default=3, 
        help="The number of input channels"
    )
    group.add_argument(
        "--num_labels", type=int, default=1000, 
        help="Number of classification labels"
    )
    group.add_argument(
        "--seq_length", type=int, default=None, 
        help="Sequence length. (Added for compatibility with profiler)"
    )
    return parser

def layernum_arg_names():
    return ['num_hidden_layers']

def seqlen_arg_names():
    return ['seq_length']