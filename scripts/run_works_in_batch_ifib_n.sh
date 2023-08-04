#!/bin/bash
#SBATCH --job-name=gpu-ifib_n
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --qos=normal

model_name=cifib
gpu=0
script_type=train

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset syn \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset citibike \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset covid19 \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset earthquakes \
        --model $model_name \
        --GPU $gpu

script_type=plot

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset syn \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset citibike \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset covid19 \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset earthquakes \
        --model $model_name \
        --GPU $gpu