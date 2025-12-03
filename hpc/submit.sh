#!/bin/sh
# $1 is path to folder with data 
# $2 is the type of dataset: adult, usa_hosue

if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

declare -a methods=("lime" "shap")

JOB_SUB="./job_submission.sh"
export JOB_PATH="~/explanation"
export DATASET="$2"

function sub {
  for i in "${methods[@]}"; do
    rm $JOB_SUB -f
    echo "Processing method: **$i** on **$1**"
    export JOB_NAME="${1//\//-}-$i-$(date +%H%M)"
    export LSB_NCPU="1"
    export LSB_MEM="8GB"
    export LSB_TIME_H="24"
    if [ "$i" == "shap" ]; then
      export LSB_MEM="500MB"
    elif [ "$i" == "lime" ]; then
      export LSB_MEM="12GB"
    fi

    export METHOD="$i"
    export DATA_PATH="$1"
    export DATA_OUT_PATH="out/$METHOD/$1"
    envsubst < "./LSF_submit.sh" >> $JOB_SUB

    bsub < $JOB_SUB
  done
}

for item in "$1"*; do
    if [ -e "$item" ] || [ -L "$item" ]; then
        if [ -d "$item" ]; then
          directory="$item"
          for file in "$directory"/*; do 
            sub "$file"
          done
        else 
          file="$item"
          sub "$file"
        fi
    fi
done
