# Auxilary Guided Synthesis
Auxilary guided synthesis is an approach to augmenting program libraries to improve problem-solving in the Program Synthesis domain. Project for solving SyGuS tasks using bee search and bustle-like model.

## Authors: 
Saqib Ameen, Thirupathi Reddy, and Habib Rahman

# Usage

## Docker image build

```docker build -t <give_an_image_name> . ```


## Check check and run image

```
docker images

docker run <give_an_image_name>
```

## Running from directory

`bee.py` contains code for ags-bee-search and `bus.py` contains code for BUS. Both work with following signature however create different log files: `bee-search.log` and `bus.log` respectively.

```sh
# Go to src dir
python3 bee.py 57 0
# Generic syntax: bee.py [TaskID] [Easy|Hard]
```

Here, `0 = easy, 1 = hard`. `TaskID` is the SyGuS task number, all tasks names are listed in `config/sygus_string_benchmarks.txt` and actual tasks are in `sygus_string_tasks/`

Running a task will create a log file in logs folder named `bee-search.log`. For the above mentioned task it will have logs like:

```
...
[Task: 56] Benchmark: exceljet1.sl
[Task: 56] Result: Success
[Task: 56] Program: _arg_1.Substr((_arg_1.IndexOf("_") + 1),_arg_1.Length())
[Task: 56] Number of evaluations: 20705
[Task: 56] 2023-02-13 15:44:44.304960
[Task: 56] Time taken: 0:00:02.525268
...
```

## src

It is the source code directory and contains all the source code in Python for running bee-search with Wu cost fn.

## models

Contains pre-trained models

## config

It contains the benchmark and properties configuration
