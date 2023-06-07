#!/bin/bash
#
export OMP_NUM_THREADS=1

if [ "$#" -eq 0 ]; then
    world_size=40
else
    world_size=$1
fi

python src/main.py \
    --world-size $world_size \
    --mode "test" \
    --agent Levin \
    --problemset-path problems/sokoban_problems.json \
    --expansion-budget 2000 \
    --seed 1 \
    --wandb-mode disabled \
    --model-path runs/Sokoban-april-train_Levin-e2000-t300_313_1684775448 \
    --model-suffix  "best_expanded" \
