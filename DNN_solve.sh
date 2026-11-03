#!/bin/bash
#SBATCH --job-name=puppet_solve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --gres=gpu:v100
#SBATCH --time=24:00:00
#SBATCH --output=%jpuppet.out

source /cluster/$USER/venvs/puppet-dca/bin/activate

cd ~/puppet-cube-dca

echo "Node:"
hostname

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo
echo "PyTorch:"
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'

echo
echo "Solve States in Dataset..."

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python -u search_methods/astar.py \
--states data/puppet/test/scrambles \
--model_dir saved_models/puppet_smart_big/current/  \
--env puppet \
--weight 0.6 \
--batch_size 10000 \
--results_dir results/puppet/ \
--language cpp
#--verbose
#--verbose | grep --line-buffered -v -E "Iteration|Times" | grep --line-buffered "."
# 2000 - 800 - 1000(dumb)
#env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
#python -u search_methods/gbfs.py \
#--model_dir saved_models/puppet_r8/current/  \
#--num_states 50 \
#--back_max 20 \
#--env puppet \
#--max_steps 50
echo
echo "Finished." 
