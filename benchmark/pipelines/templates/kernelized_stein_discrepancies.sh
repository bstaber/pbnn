#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J ksd
#SBATCH -o ksd_%j.o
#SBATCH -e ksd_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

python pipelines/kernelized_stein_discrepancies.py --algorithm={{algorithm}} --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --savedir={{savedir}}