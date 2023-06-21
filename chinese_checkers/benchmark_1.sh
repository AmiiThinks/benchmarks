#!/bin/bash

# Benchmark for parallel self_play games for 20 iterations

# start the parallel_play in the background
python3 $PWD/python2/parallel/jax_async_play.py &


