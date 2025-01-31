#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J lengthscale
#SBATCH -o lengthscale_%j.o
#SBATCH -e lengthscale_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

python pipelines/lengthscale_estimation.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}}