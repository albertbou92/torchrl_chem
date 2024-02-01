#!/bin/bash

#SBATCH --job-name=reinvent
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/reinvent%j.txt
#SBATCH --error=slurm_errors/reinvent%j.txt

current_commit=$(git rev-parse --short HEAD)
project_name="acegen-open-example-check-$current_commit"
agent_name="reinvent"

export PYTHONPATH=$(dirname $(dirname $PWD))
python $PYTHONPATH/examples/reinvent/reinvent.py \
  logger_backend=wandb \
  experiment_name="$project_name" \
  agent_name="$agent_name" \
  molscore=MolOpt \
  molscore_include=[Albuterol_similarity]

# Capture the exit status of the Python command
exit_status=$?
# Write the exit status to a file
if [ $exit_status -eq 0 ]; then
  echo "${group_name}_${SLURM_JOB_ID}=success" >>> report.log
else
  echo "${group_name}_${SLURM_JOB_ID}=error" >>> report.log
fi