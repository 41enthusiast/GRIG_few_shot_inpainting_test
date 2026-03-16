#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu:turing:1
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=long-high-prio # Use the short QOS
#SBATCH -t 7-0 # Set maximum walltime to 1 day
#SBATCH --job-name=wd_inpainting_test # Name of the job
#SBATCH --mem=16G # Request 16Gb of memory

#SBATCH -o outputs/program_output.txt
#SBATCH -e outputs/whoopsies.txt

# Load the global bash profile
source /etc/profile
module unload cuda
module load cuda/11.8
export PATH=$PATH:$(python3.11 -m site --user-base)/bin

# Load your Python environment
source grig/bin/activate
python3.11 train.py --path ../data/art_painting/train --test_path ../data/art_painting/val --im_size 512 --batch_size 8 --aug False --efficient_net pre_train/tf_efficientnet_lite0-0aa007d2.pth