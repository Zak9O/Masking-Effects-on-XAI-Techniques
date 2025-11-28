#!/bin/bash

OUT="./copied_to_cluster"
rm -rf $OUT
mkdir $OUT

cp ./README.md ./LSF_submit.sh ./setup.sh ./submit.sh $OUT
cp ./pyp.toml $OUT/pyproject.toml

cp -r ./scripts/ $OUT
cp -r ../hierarchies/ $OUT/hierarchies

scp -i ~/.ssh/id_ed25519 -r $OUT s225169@transfer.gbar.dtu.dk:
