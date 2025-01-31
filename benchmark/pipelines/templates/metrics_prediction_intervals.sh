#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J metrics
#SBATCH -o metrics_%j.o
#SBATCH -e metrics_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

python pipelines/metrics_prediction_intervals.py --algorithm={{algorithm}} --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --savedir={{savedir}}