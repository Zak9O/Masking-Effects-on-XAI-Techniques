echo "=========================================================="
echo "Job started on $(date)"
echo "Running on host $(hostname)"
echo "Using $LSB_NCPU cores"

echo "=========================================================="

source ./common_vars.sh

module load python3/3.12.11
source "$HPC/.venv/bin/activate"
echo "Running Python script..."

python3 scripts/importance_vector.py \
  "$HPC/${DATA_PATH}" \
  "$HPC/out/${METHOD}/${DATA_PATH}" \
  "${METHOD}"

echo "=========================================================="
echo "Job finished on $(date)"
echo "=========================================================="
