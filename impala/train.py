from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.logger import pretty_print
import gym
import time
import tqdm
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model_type", type=str, default='LSTM', help="Model type to use: FF, LSTM, TRF")
    args = parser.parse_args()
    
    
    config = ImpalaConfig()
    config = config.training(lr=0.0005, train_batch_size=500)
    config = config.resources(num_gpus=args.num_gpus)
    
    config = config.rollouts(num_rollout_workers=32,rollout_fragment_length=50)  
    config = config.framework("torch")
    config['clip_rewards'] = True
    if args.model_type=='LSTM':
        config['model']['use_lstm']=True
        config['model']['lstm_cell_size']=2048
    elif args.model_type=='TRF':
        config['model']['use_attention']=True
        config['model']['attention_num_transformer_units']= 12
        config['model']['attention_dim']= 512
        config['model']['attention_num_heads']: 8
        config['model']['attention_head_dim']= 64
        config['model']['attention_memory_inference']=256
        config['model']['attention_memory_training']= 256 
        config['model']['attention_position_wise_mlp_dim']= 512
    config = config.environment(env='ALE/SpaceInvaders-v5')
    print(config.to_dict())  
    # Build an Algorithm object from the config and run 1 training iteration.
    algo = config.build()  
    
    start_time = time.time()
    
    iteration_times = []
    
    for i in tqdm.tqdm(range(30)):
        iteration_start_time = time.time()
        result = algo.train()
        print('Steps trained',result['num_agent_steps_trained'])
        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        iteration_times.append(iteration_time)
    
    end_time = time.time()
    execution_time = end_time - start_time
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    
    print("Total execution time:", execution_time, "seconds")
    print("Average iteration time:", avg_iteration_time, "seconds")
