#!/bin/bash

for ((aug = 0; aug <= 0; aug++))
	do
	for ((model = 2; model <= 2; model ++))
		do
		for ((taskId = 1; taskId <=89; taskId ++))
			do
				if [[ $taskId -eq 17 || $taskId -eq 24 || $taskId -eq 71 || $taskId -eq 83 || $taskId -eq 84 || $taskId -eq 88 || $taskId -eq 89 ]]; then
					echo "Augment : ${aug}"
					echo "Model : ${model}"
					echo "TaskID : $taskId"
					sbatch --export=m="${model}",t="${taskId}",a="${aug}" batcher.sh
				fi
			done
		done
	done
