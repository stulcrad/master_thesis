echo "initializing Python environment..."
uname -a

ml Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.9.1

source ~/virtualenvs/Master_thesis/bin/activate

echo "done"