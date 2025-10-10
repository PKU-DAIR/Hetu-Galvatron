import os
from galvatron.core import initialize_galvatron
from galvatron.core.cost_model import GalvatronCostModelHandler
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name

if __name__ ==  '__main__':
    args = initialize_galvatron(model_args, mode="cost_model")
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    cost_model_handler = GalvatronCostModelHandler(args)
    cost_model_handler.set_cost_model_handler_info(path, model_layer_configs(config), model_name(config))
    cost_model_handler.initialize_cost_model_handler()

    cases = [
        {"global_batch_size": 32, "chunks": 8, "strategy": [1, 2, 4, {'fsdp':0}]},
        {"global_batch_size": 64, "chunks": 4, "strategy": [1, 2, 4, {'fsdp':0}]},
        {"global_batch_size": 64, "chunks": 4, "strategy": [1, 2, 4, {'fsdp':1}]},
        {"global_batch_size": 64, "chunks": 4, "strategy": [1, 2, 4, {}]},
    ]
    
    for case in cases:
        global_batch_size = case["global_batch_size"]
        chunks = case["chunks"]
        strategy = case["strategy"]
        print(f"\n=== Check Cost for Global_batch_size: {global_batch_size}, Chunks: {chunks}, Strategy: {strategy} ===")
        time_cost = cost_model_handler.get_time_cost_for_specific_strategy(strategy, global_batch_size, chunks)
        memory_cost = cost_model_handler.get_memory_cost_for_specific_strategy(strategy, global_batch_size, chunks)
        print(f'Time Cost: {time_cost}')
        print(f'Memory Cost: {memory_cost}')