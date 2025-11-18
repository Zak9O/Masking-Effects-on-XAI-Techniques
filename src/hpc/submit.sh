#!/bin/sh

if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

source ./common_vars.sh

declare -a methods=("shap") #("t_closeness" "alpha_k_anonymity" "l_diversity" "k_anonymity")

function sub {
  for i in "${methods[@]}"; do
    rm tmp_submit_file.sh -f
    echo "Processing method: **$i** on **$1**"

    export JOB_NAME="$i"
    if [ "$i" == "shap" ]; then
      export LSB_NCPU="1"
      export LSB_MEM="500MB"
    else 
      export LSB_NCPU="1"
      export LSB_MEM="8GB"
    fi

    envsubst < "./LSF_options.sh" >> "tmp_submit_file.sh"

    export METHOD="$i"
    export DATA_PATH="$1"
    envsubst < "./LSF_run.sh" >> "tmp_submit_file.sh"

    bsub < tmp_submit_file.sh
  done
}

for item in "$(basename "$1")"/*; do
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
