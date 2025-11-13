#!/bin/sh

source ./common_vars.sh

rm tmp_submit_file.sh -f
ANONYMIZATION_METHODS="t_closeness alpha_k_anonymity l_diversity k_anonymity"

declare -a methods=("t_closeness" "alpha_k_anonymity" "l_diversity" "k_anonymity")

for i in "${methods[@]}"; do
# for METHOD in $ANONYMIZATION_METHODS; do
  echo "Processing method: **$i**"
  sed "s/JOB_NAME/$i/g" "./LSF_options.sh" >> "tmp_submit_file.sh"
  sed "s/CHANGE_THIS_METHOD/$i/g" "./LSP_run.sh" >> "tmp_submit_file.sh"

  # bsub < tmp_submit_file.sh
  rm tmp_submit_file.sh -f
done

