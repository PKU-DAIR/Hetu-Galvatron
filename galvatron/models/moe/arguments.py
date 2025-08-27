def model_args(parser):
    group = parser.add_argument_group(title="Model Arguments")

    group.add_argument(
        "--model_size",
        type=str,
        default="mixtral-8x7b",
        help="Model size.",
    )
    group.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="Hidden size of transformer model.",
    )
    group.add_argument(
        "--intermediate_size",
        type=int,
        default=14336,
        help="Intermediate size of transformer model.",
    )
    group.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help="Head dimension of transformer model.",
    )
    group.add_argument(
        "--seq_length", 
        type=int, 
        default=4096, 
        help="Maximum sequence length."
    )
    group.add_argument(
        "--num_attention_heads",
        type=int,
        default=32,
        help="Number of attention heads",
    )
    group.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Number of experts per token",
    )
    group.add_argument(
        "--num_hidden_layers",
        type=int,
        default=32,
        help="Number of hidden layers",
    )
    group.add_argument(
        "--num_key_value_heads",
        type=int,
        default=8,
        help="Number of key value heads",
    )
    group.add_argument(
        "--num_local_experts",
        type=int,
        default=8,
        help="Number of local experts",
    )
    group.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Total number of vocab",
    )
    group.add_argument(
        "--rms_norm_eps",
        type=float,
        default=1e-5,
        help="RMS norm epsilon",
    )
    group.add_argument(
        "--rope_theta",
        type=float,
        default=1000000.0,
        help="Rope theta",
    )
    group.add_argument(
        "--router_aux_loss_coef",
        type=float,
        default=0.02,
        help="Router aux loss coefficient",
    )
    group.add_argument(
        "--set_aux_loss_manually",
        action="store_true",
        help="Set aux loss manually",
    )
    return parser


def layernum_arg_names():
    return ["num_hidden_layers"]
