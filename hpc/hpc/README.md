# What is this?
This folder is meant to be self contained and it should be able to run on DTU's HPC cluster.
# How to run it on the HPC cluster?
Simply run `setup.sh` followed by `submit.sh
# Extra causion with LIME
I have manually edited the downloaded LIME files. The same must be done on the HPC cluster for it to work. The `top_label` functionality of the `submoduler_pick.py` module in the `LIME` folder in the `.venv/lib` folder must be removed. It is no more than 3 lines or so

