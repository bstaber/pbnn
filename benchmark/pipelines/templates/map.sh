#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J map
#SBATCH -o map_%j.o
#SBATCH -e map_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

python pipelines/map.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --num_datasets={{num_datasets}}