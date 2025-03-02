#!/usr/bin/env bash
#SBATCH -J naglmbis-refit
#SBATCH -p cpu
#SBATCH -t 96:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --output slurm-%x.%A.out

PORT=8002

source ~/.bashrc

ENVFILE="evaluator-0-4-10"

conda activate $ENVFILE

# write force field
# python set-up-forcefield.py

# write client options
# python setup-options.py --port $PORT

# run fit
python execute-fit-slurm-distributed.py                 \
    --port                  $PORT                       \
    --n-min-workers         1                           \
    --n-max-workers         60                          \
    --memory-per-worker     8                           \
    --walltime              "08:00:00"                  \
    --queue                 "gpu"                       \
    --conda-env             $ENVFILE                    \
    --extra-script-option   "--gpus-per-task=1"              # note: this is Iris specific
