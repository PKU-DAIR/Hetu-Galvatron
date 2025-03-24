from galvatron.core import initialize_galvatron, GalvatronSearchEngine
from galvatron.models.vit_hf.arguments import model_args
from galvatron.models.vit_hf.ViTModel_hybrid_parallel import get_vit_config
from galvatron.models.vit_hf.meta_configs import model_name, model_layer_configs
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    
    config = get_vit_config(args)
    
    path = os.path.dirname(os.path.abspath(__file__))
    
    print(args)
    print(config)
    
    search_engine = GalvatronSearchEngine(args)
    
    search_engine.set_search_engine_info(
        path, 
        model_layer_configs(config), 
        model_name(config)
    )
    
    search_engine.set_model_type('vit')
    
    search_engine.initialize_search_engine()
    
    search_engine.parallelism_optimization()