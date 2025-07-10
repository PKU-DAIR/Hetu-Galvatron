import os
import logging

def ensure_log_dir(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_thread_logger(bsz, chunk, min_tp, max_tp, vsp, embed_sdp, log_dir='logs'):

    logger_name = f"galvatron_bsz{bsz}_chunk{chunk}_min_tp{min_tp}_max_tp{max_tp}_vsp{vsp}_embed_sdp{embed_sdp}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    log_dir = os.path.join(log_dir, f"search_bsz{bsz}_chunk{chunk}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"min_tp{min_tp}_max_tp{max_tp}_vsp{vsp}_embed_sdp{embed_sdp}.log")
    file_handler = logging.FileHandler(log_file, mode='w')

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger