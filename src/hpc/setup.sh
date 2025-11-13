source ./common_vars.sh

python -m venv "$HPC/.venv"
source "$HPC/.venv/bin/activate"

pip install -r "$HPC/requirements.txt"

deactivate
