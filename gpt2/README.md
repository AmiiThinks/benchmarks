# GPT benchmark

This is a benchmark of training GPT-2 with 200 updates.

## Environment setup


```bash
# CUDA 11.8 or above is required.
# python 3.8 or above is required.

# insetall pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# get hugginface Transformer 
git clone git@github.com:huggingface/transformers.git
cd transformers
pip3 install .
cd ..
```
## Hardware Info
```
2 x Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz
4 x  NVIDIA RTX A5000
RAM: 378G (DDR4 3200MHz)
```


## Run benchmark
```bash
rm -r /tmp/test-clm; LOCAL_RANK=0,1,2,3; CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node 4 --use-env examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --fp16 --max_steps 200
```


## Result 
```
epoch                    =       1.38
train_loss               =     3.2569
train_runtime            = 5:12:44.54
train_samples            =       2318
train_samples_per_second =      0.171
train_steps_per_second   =      0.011
```

