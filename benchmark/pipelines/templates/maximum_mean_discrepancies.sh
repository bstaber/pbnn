#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J mmd
#SBATCH -o mmd_%j.o
#SBATCH -e mmd_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

python pipelines/maximum_mean_discrepancies.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --savedir={{savedir}} --dataset_idx=0