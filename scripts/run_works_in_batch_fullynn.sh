#!/bin/bash
#SBATCH --job-name=gpu-fullynn
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --qos=normal

model_name=fullynn
gpu=0

script_type=train

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset synthetic \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset bookorder \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset mooc \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset retweet \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset stackoverflow \
        --model $model_name \
        --GPU $gpu

script_type=plot

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset synthetic \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset bookorder \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset mooc \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset retweet \
        --model $model_name \
        --GPU $gpu

python3 ../batch_task_worker.py \
        --procedure_name TPP \
        --script_type $script_type \
        --dataset stackoverflow \
        --model $model_name \
        --GPU $gpu