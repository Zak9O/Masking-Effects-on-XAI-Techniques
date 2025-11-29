#!/bin/sh
# $1 should be the path to the file that should be anonymized
# $2 should be the name of the dataset we consider i.e. adult, usa_house
# $3 should be the k used for k anonymity

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <path_to_file> <dataset_name> <k_value>"
  exit 1
fi

rm tmp_submit_file.sh -f

declare -a methods=("k_anonymity" "t_closeness" "alpha_k_anonymity" "l_diversity")

JOB_SUB="./job_submission.sh"
export JOB_PATH="~/anonymization"

for i in "${methods[@]}"; do
  rm $JOB_SUB -f
  echo "Processing method: **$i**"
  export JOB_NAME="$2-$i-$(date +%H%M)"
  export LSB_NCPU="1"
  export LSB_MEM="1GB"
  export LSB_TIME_H="24"

  export DATA_PATH=$1
  export DATASET=$2
  export HIERARCHIES_PATH="$JOB_PATH/hierarchies/$DATASET"
  export SAVE_DIR_PATH="$JOB_PATH/out/$DATASET/$i"
  export ANONYMIZATION_METHOD=$i
  export K=$3

  envsubst < "./LSF_submit.sh" >> $JOB_SUB

  bsub < $JOB_SUB
done

