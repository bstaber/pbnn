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

{% if algorithm == "mcdropout" %}
python pipelines/mcdropout.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --dropout={{dropout}} --num_datasets={{num_datasets}}
{% elif algorithm == "hmc" %}
python pipelines/hmc.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --dataset_idx={{dataset_idx}} --num_datasets={{num_datasets}}
{% elif algorithm == "swag" %}
python pipelines/swag.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --num_datasets={{num_datasets}}
{% elif algorithm == "deep_ensembles" %}
python pipelines/deep_ensembles.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --dataset_idx={{dataset_idx}} --num_datasets={{num_datasets}}
{% elif algorithm == "laplace" %}
python pipelines/laplace_pretrained.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --num_datasets={{num_datasets}}
{% elif algorithm == "sghmc_svrg" %}
python pipelines/sgmcmc_single_dataset.py --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --dataset_idx={{dataset_idx}} --algorithm={{algorithm}} --num_datasets={{num_datasets}} --init_method={{init_method}}
{% else %}
python pipelines/sgmcmc.py --algorithm={{algorithm}} --experiment={{experiment}} --seed={{seed}} --workdir={{workdir}} --step_size={{step_size}} --num_datasets={{num_datasets}} --init_method={{init_method}}
{% endif %}
