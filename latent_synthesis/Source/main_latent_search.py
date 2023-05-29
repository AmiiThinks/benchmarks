import time
import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls


if __name__ == '__main__':
    
    start_time = time.time()
    
    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    
    params = torch.load(Config.model_params_path, map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task_cls, dsl)
    
    StdoutLogger.log('Main', f'Starting CEM with model {Config.model_name} for task {Config.env_task}')
    
    best_program, converged, num_evaluations = searcher.search()
    
    StdoutLogger.log('Main', f'Converged: {converged}')
    StdoutLogger.log('Main', f'Final program: {best_program}')
    StdoutLogger.log('Main', f'Number of evaluations: {num_evaluations}')
    
    StdoutLogger.log('Main', f'CEM finished in {time.time() - start_time} seconds')
