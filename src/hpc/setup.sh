source ./common_vars.sh

python -m venv "$HPC/.venv"
source "$HPC/.venv/bin/activate"

pip install -r "$HPC/requirements.txt"

mkdir job_out out 
mkdir out/t_closeness out/alpha_k_anonymity out/l_diversity out/k_anonymity

deactivate
