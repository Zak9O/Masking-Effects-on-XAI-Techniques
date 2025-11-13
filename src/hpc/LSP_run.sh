echo "=========================================================="
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Using $LSB_NCPU cores"

echo "=========================================================="

source ./common_vars.sh

module load python3/3.12.11
source "$HPC/.venv/bin/activate"
echo "Running Python script..."

# Can be "t_closeness", "alpha_k_anonymity", "l_diversity", or "k_anonymity"
local anonymization_method="CHANGE_THIS_METHOD"
local data_path="$HPC/clean_data.csv"
local hierarchies_path="$HPC/hierarchies"
local save_dir_path="$MY_HOME/hpcout/${anonymization_method}"

python3 anonymize.py \
  --anonymization_method ${anonymization_method} \
  --data_path ${data_path} \
  --hierarchies_path ${hierarchies_path} \
  --save_dir_path ${save_dir_path}

echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="
