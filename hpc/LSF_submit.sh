#!/bin/sh
### General LSF options
# --- Specify the queue --
#BSUB -q hpc
# --- Set the job Name --
#BSUB -J ${JOB_NAME}
# --- Set usage info --
#BSUB -n ${LSB_NCPU}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${LSB_MEM}]"
# --- Set walltime limit: hh:mm --
#BSUB -W ${LSB_TIME_H}:00
# --- Specify the output and error file. %J is the job-id --
#BSUB -o logs/${JOB_NAME}.out
#BSUB -e logs/${JOB_NAME}.err

# --- End of LSF options --
echo "=========================================================="
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Using $LSB_NCPU cores"

echo "=========================================================="

module load python3/3.12.11

echo "Running Python script..."
uv run scripts/anonymize.py \
  ${DATA_PATH} \
  ${HIERARCHIES_PATH} \
  ${SAVE_DIR_PATH} \
  ${ANONYMIZATION_METHOD} \
  ${K} \
  ${DATASET}

echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="
