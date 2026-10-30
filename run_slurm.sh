#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_%j.log
#SBATCH --exclude=ragpu[003,006,004],racn115,racn116

echo "Running on node: $(hostname)"
mkdir -p logs

# Activate Conda Environment
module load MLDL/miniconda3 2>/dev/null || true
if [ -d "/home/chinmayb_iitp/FY/conda_env_v2" ]; then
    conda activate /home/chinmayb_iitp/FY/conda_env_v2
elif [ -d "/home/chinmayb_iitp/Pharmacovigilance/conda_env_v2" ]; then
    conda activate /home/chinmayb_iitp/Pharmacovigilance/conda_env_v2
fi

# Run the pipeline
bash run_all.sh
