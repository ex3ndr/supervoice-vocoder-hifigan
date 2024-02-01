set -e
export PYTORCH_ENABLE_MPS_FALLBACK=1
accelerate launch ./train.py