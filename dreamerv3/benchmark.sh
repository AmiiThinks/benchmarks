#!/bin/bash
echo "Beginning atari benchmark."
{ time python3 train.py --logdir ./logs --configs atari --run.steps 5000 ; }  &>> ./dreamer_output.log
rm -rf ./logs
echo "Beginning dmc benchmark."
{ time python3 train.py --logdir ./logs --configs dmc_vision --run.steps 5000 ; }  &>> ./dreamer_output.log
rm -rf ./logs
echo "Beginning crafter benchmark."
{ time python3 train.py --logdir ./logs --configs crafter --run.steps 2000 ; } &>> ./dreamer_output.log

