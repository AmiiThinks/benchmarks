# Benchmark for MicroRT 

Author: Rubens O. Moraes (rubensolv@gmail.com)

## Environments:
We provided different environments for MicroRTS, according to the map and enemies for each map. Each environment is related with one different folder in the repository.

## Building Docker image for specific maps:
It requires getting into the folders. Example for folder map8x8Dockerfile. 
```bash
sudo docker build . -t map8x8dockerfile/microrts:1.0
```
### Following by the execution:
```bash
sudo docker run -it map8x8dockerfile/microrts:1.0
```
## Benchmarks
Each container is going to run and produce a output as described below:

> col_SP_910000_5355077.txt;col_FP_910000_5512262.txt;col_CS_910000_5421331.txt;col_SP_910000_5512217.txt;col_DO_910000_5355082.txt;WorkerRush(AStarPathFinding);col_DO_910000_5354876.txt;col_CS_910000_5374685.txt;col_FP_910000_5355052.txt;
SP_910000_5355077.txt;0.5;0.25;0.0;1.0;0.0;0.0;1.0;0.0;0.25;
CS_910000_5374685.txt;1.0;0.75;0.25;1.0;0.5;1.0;0.5;0.5;0.5;
FP_910000_5355052.txt;0.75;0.75;0.25;0.75;0.75;0.25;0.5;0.5;0.5;
CS_910000_5421331.txt;1.0;0.5;0.5;1.0;0.25;1.0;0.5;0.75;0.75;
DO_910000_5355082.txt;1.0;1.0;0.75;1.0;0.5;1.0;0.5;0.5;0.25;
FP_910000_5512262.txt;0.75;0.5;0.5;0.5;0.0;0.0;0.5;0.25;0.25;
SP_910000_5512217.txt;0.0;0.5;0.0;0.5;0.0;0.5;0.25;0.0;0.25;
DO_910000_5354876.txt;0.0;0.5;0.5;0.75;0.5;0.0;0.5;0.5;0.5;Finished all battles Mon May 29 21:14:11 GMT 2023
Took: 8 seconds

It also produce a file in the container /home/MicroRTS/log.txt. That file could be amazing to be copied for validations.
