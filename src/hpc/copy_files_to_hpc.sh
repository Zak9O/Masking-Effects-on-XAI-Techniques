#!/bin/bash

source ./common_vars.sh

rm -rf ./hpc
mkdir ./hpc

cp ./common_vars.sh ./LSF_options.sh ./LSF_run.sh ./README.md ./requirements.txt ./setup.sh ./submit.sh ./hpc/

cp -r ./scripts/ ./hierarchies/ ./hpc

scp -i ~/.ssh/id_ed25519 -r hpc s225169@transfer.gbar.dtu.dk:

# rm -rf ./hpc
