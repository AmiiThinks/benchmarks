#!/bin/bash
#SBATCH --account=def-lelis
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --gres=gpu:1
#SBATCH --mem=64G        # memory per node
#SBATCH --time=00-24:00      # time (DD-HH:MM)
#SBATCH --output=log/%N-%j.out  # %N for node name, %j for jobID

module load python/3 cuda cudnn
source tensorflow/bin/activate
cd src/
python3 bee.py -t ${t} -d 0 -l ${t}_${a}_${m}.log -m bustle_model_0${m}.hdf5 -b bustle_benchmarks.txt -a "${a}" -p 14000000
