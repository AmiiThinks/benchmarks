#!/bin/bash

source latent_synthesis_env/bin/activate
python3 Source/main_latent_search.py \
    --experiment_name leaps_cem_benchmark \
    --env_task FourCorners \
    --model_name LeapsVAE \
    --model_hidden_size 256 \
    --model_params_path Source/params/leaps_vae_256.ptp \
    --model_seed 1 \
    --search_population_size 512 \
    --data_max_demo_length 1000 \
    --search_number_executions 64 \
    --search_number_iterations 50 \
    --multiprocessing_active \
    --disable_gpu
