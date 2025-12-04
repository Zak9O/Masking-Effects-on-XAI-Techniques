#!/bin/bash

OUT="./anonymization"
rm -rf $OUT
mkdir $OUT

cp ./README.md ./LSF_submit.sh ./submit.sh $OUT
cp ./pyp.toml $OUT/pyproject.toml

cp -r ./scripts/ $OUT

if [ "$1" = "all" ]; then
  mkdir $OUT/data
  cp ../data/adult/clean.csv $OUT/data/adult.csv
  cp ../data/usa_house/clean.csv $OUT/data/usa_house.csv
  cp ../data/cervic_cancer/clean.csv $OUT/data/cervic_cancer.csv
  cp -r ../hierarchies/ $OUT/hierarchies
fi

mkdir $OUT/logs

scp -i ~/.ssh/id_ed25519 -r $OUT s225169@transfer.gbar.dtu.dk:
