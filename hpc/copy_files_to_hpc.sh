#!/bin/bash

OUT="./explanation"
rm -rf $OUT
mkdir $OUT

cp ./README.md ./LSF_submit.sh ./submit.sh $OUT
cp ./pyp.toml $OUT/pyproject.toml

cp -r ./scripts/ $OUT

if [ "$1" = "all" ]; then
  mkdir $OUT/data
  cp -r ../data/adult $OUT/data
  cp -r ../data/usa_house $OUT/data
  cp -r ../hierarchies/ $OUT
fi

mkdir $OUT/logs

scp -i ~/.ssh/id_ed25519 -r $OUT s225169@transfer.gbar.dtu.dk:
