#!/bin/bash

OUT="./explanation"
rm -rf $OUT
mkdir $OUT

cp ./README.md ./LSF_submit.sh ./submit.sh $OUT
cp ./pyp.toml $OUT/pyproject.toml

cp -r ./scripts/ $OUT

if [ "$1" = "all" ]; then
  mkdir $OUT/data
    datasets=("adult" "usa_house" "cervic_cancer")
    anonymity_types=("alpha_k_anonymity" "k_anonymity" "l_diversity" "t_closeness")

    for dataset in "${datasets[@]}"; do
      mkdir -p $OUT/data/$dataset
      for anon_type in "${anonymity_types[@]}"; do
        cp -r ../data/$dataset/$anon_type $OUT/data/$dataset/
      done
    done
  cp -r ../hierarchies/ $OUT
fi

mkdir $OUT/logs

scp -i ~/.ssh/id_ed25519 -r $OUT s225169@transfer.gbar.dtu.dk:
