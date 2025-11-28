#!/bin/sh
### General LSF options
# --- Specify the queue --
#BSUB -q hpc
# --- Set the job Name --
#BSUB -J JOB_NAME
# --- Ask for number of cores (default: 1) --
#BSUB -n 1
# --- Specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=1GB]"

# --- Set walltime limit: hh:mm --
#BSUB -W 8:00


# --- Specify the output and error file. %J is the job-id --
#BSUB -o job_out/JOB_NAME_%J.out
#BSUB -e job_out/JOB_NAME_%J.err

# --- End of LSF options --
