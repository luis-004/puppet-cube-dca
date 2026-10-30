#!/bin/bash
#SBATCH --job-name=puppet_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --output=outs/%jpuppet.out
source /cluster/$USER/venvs/puppet-dca/bin/activate

cd ~/puppet-cube-dca # path may not be right

echo "Node:"
hostname

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo
echo "PyTorch:"
python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'

echo
echo "Starting AVI..."

START_TIME=$(date +%s)

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python ctg_approx/avi.py \
    --env puppet \
    --states_per_update 50000000 \
    --batch_size 10000 \
    --epochs_per_update 1 \
    --nnet_name puppet \
    --max_itrs 1000000 \
    --loss_thresh 0.06 \
    --back_max 60 \
    --num_update_procs 8 \
    --update_nnet_batch_size 10000 \
    --update_method=GBFS \
    --single_gpu_training

echo
echo "Finished." 

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "Finished."
echo "Training time: ${ELAPSED} seconds"
printf 'Training time: %02d:%02d:%02d\n' \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))


# sacct -j JOBID --format=JobID,Elapsed,State

#PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
#python ctg_approx/avi.py \
#    --env puppet \
#    --states_per_update 50000000 \
#    --batch_size 10000 \
#    --epochs_per_update 1 \
#    --nnet_name puppet_giga \
#    --max_itrs 1000000 \
#    --loss_thresh 0.06 \
#    --back_max 60 \
#    --num_update_procs 8 \
#    --update_nnet_batch_size 10000 \
#    --max_update_steps 1 \
#    --single_gpu_training

#PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
#python ctg_approx/avi.py \
#    --env puppet \
#    --states_per_update 5000000 \
#    --batch_size 1000 \
#    --epochs_per_update 1 \
#    --nnet_name puppet_mega \
#    --max_itrs 1000000 \
#    --loss_thresh 0.06 \
#    --back_max 30 \
#    --num_update_procs 8 \
#    --update_nnet_batch_size 10000 \
#    --max_update_steps 1 \
#    --single_gpu_training

