echo "initializing Python environment..."
uname -a

export PYTHONPATH=~/master_thesis/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

ml Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source ~/virtualenvs/Master_thesis/bin/activate

echo "done"