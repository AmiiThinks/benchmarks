#!/bin/bash

source latent_synthesis_env/bin/activate
python3 Source/main_trainer.py \
    --experiment_name leaps_vae_benchmark \
    --model_name LeapsVAE \
    --data_class_name ProgramsAndDemosDataset \
    --data_program_dataset_path Source/data/reduced_programs_dataset.pkl \
    --model_hidden_size 256 \
    --data_batch_size 128 \
    --data_max_demo_length 1000 \
    --trainer_num_epochs 10
