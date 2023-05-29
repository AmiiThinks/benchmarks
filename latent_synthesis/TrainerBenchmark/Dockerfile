FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
LABEL maintainer="Tales H. Carvalho"

WORKDIR /app

COPY Source/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY Source/ /app

WORKDIR /app

CMD python3 main_trainer.py \
    --experiment_name leaps_vae_benchmark \
    --model_name LeapsVAE \
    --data_class_name ProgramsAndDemosDataset \
    --data_program_dataset_path data/reduced_programs_dataset.pkl \
    --model_hidden_size 256 \
    --data_batch_size 32 \
    --data_max_demo_length 1000 \
    --trainer_num_epochs 10
