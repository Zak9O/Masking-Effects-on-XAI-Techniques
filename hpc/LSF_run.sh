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
anonymization_method="CHANGE_THIS_METHOD"
data_path="$HPC/data/clean.csv"
hierarchies_path="$HPC/hierarchies"
save_dir_path="$HPC/out/${anonymization_method}"

python3 anonymize.py \
  ${data_path} \
  ${hierarchies_path} \
  ${save_dir_path} \
  ${anonymization_method} \

echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="
