#!/bin/bash

script_name=${0##*/}
if [[ "$#" -ne 2 ]]; then
	echo " "
	echo "#############################################################################"
	echo ""
	echo " Please provide logs directory argument      "
	echo " Usage: $script_name 'logs directory' "
	echo " Example: $script_name '2409'              "
	echo ""
	echo "#############################################################################"
	echo ""
	exit 1
fi

echo "start.."

cat $1 | grep -A 6 -B 5 "Result: Success" | grep "Program:" | sed 's/^.*Program: //g' > $2"_programs.txt"
cat $1 | grep -A 6 -B 5 "Result: Success" | grep "Benchmark:" | sed 's/^.*Benchmark: //g' > $2"_benchmarks.txt"
cat $1 | grep -A 6 -B 5 "Result: Success" | grep "Result:" | sed 's/^.*Result: //g' > $2"_results.txt"
cat $1 | grep -A 6 -B 5 "Result: Success" | grep "Number of evaluations:" | sed 's/^.*Number of evaluations: //g' > $2"_evaluations.txt"
cat $1 | grep -A 6 -B 5 "Result: Success" | grep "Time taken:" | sed 's/^.*Time taken: //g' > $2"_times.txt"
