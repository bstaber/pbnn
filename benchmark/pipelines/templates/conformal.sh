#!/bin/bash

#SBATCH -p {{partition}}.q
#SBATCH -n {{n}}
{% if gpu > 0 %}
#SBATCH --gres=gpu:{{gpu}}
{% endif %}
#SBATCH -t {{t}}
#SBATCH -J {{algorithm}}
#SBATCH -o {{algorithm}}_%j.o
#SBATCH -e {{algorithm}}_%j.e

source ~/.bashrc
conda_init
conda activate jax-cipiu

{% if algorithm == "split_cqr" %}
python pipelines/split_cqr.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --algorithm={{algorithm}} --step_size={{step_size}}
{% else %}
python pipelines/mapie_conformal.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --algorithm={{algorithm}} --step_size={{step_size}}
{% endif %}