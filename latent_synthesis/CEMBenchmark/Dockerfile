FROM python:3.10
LABEL maintainer="Tales H. Carvalho"

WORKDIR /app

COPY Source/requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY Source/ /app

WORKDIR /app

CMD python3 main_latent_search.py \
    --experiment_name leaps_cem_benchmark \
    --env_task FourCorners \
    --model_name LeapsVAE \
    --model_hidden_size 256 \
    --model_params_path params/leaps_vae_256.ptp \
    --model_seed 1 \
    --search_population_size 512 \
    --data_max_demo_length 1000 \
    --search_number_executions 64 \
    --search_number_iterations 50 \
    --multiprocessing_active \
    --disable_gpu
