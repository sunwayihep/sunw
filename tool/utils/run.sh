#!/bin/bash
#SBATCH -J su6_16 
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
source env.sh

COLOR=6
NS=16
NT=16
echo COLOR=$COLOR

PYTHON=/public/home/wangjc_itp/software/conda/miniconda3/envs/pyquda/bin/python
$PYTHON run.py $COLOR $NS $NT
