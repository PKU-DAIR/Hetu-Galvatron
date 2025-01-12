from galvatron.core import initialize_galvatron, GalvatronSearchEngine
from galvatron.models.bert_hf.arguments import model_args
from galvatron.models.bert_hf.hf_configs import config_from_meta, set_model_config, model_name, model_layer_configs
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args=False)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    # search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type('bert') 
    
    search_engine.initialize_search_engine()
    # search_engine.check_cost_model(bsz=16 ,chunk=1, min_tp=1)
    search_engine.parallelism_optimization()