# FA Dyna Benchmark
#### by Edan Meyer

This benchmark uses a function approximation verison of dyna with an auxillary observation reconstruction loss term. During both rollouts and training, all model passes are done on the GPU. Because there is only one environment instance, fast CPU -> GPU communication speed should have a large effect on performance (which is pretty typical in many RL experiments).

## Run Instructions

WIP, I will add a docker file soon