#!/bin/sh
### General LSF options
# --- Specify the queue --
#BSUB -q hpc
# --- Set the job Name --
#BSUB -J ${JOB_NAME}
# --- Ask for number of cores (default: 1) --
#BSUB -n 1
# --- Specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=8GB]"

# --- Set walltime limit: hh:mm --
# NOTE: Your LIME calculation with 5000 explanations might take a long time.
# I've set this to 1 hour (1:00) as a test.
# You may need to increase this significantly (e.g., to 24:00 or more).
#BSUB -W 8:00


# --- Specify the output and error file. %J is the job-id --
#BSUB -o job_out/${JOB_NAME}_%J.out
#BSUB -e job_out/${JOB_NAME}_%J.err

# --- End of LSF options --
