export EXPERT_SLICING=$1
export EP_SIZE=$2
if [ $EXPERT_SLICING -eq 1 ]; then
    echo "Setting EXPERT_SLICING=1, EP_SIZE=$2"
    deepspeed --num_gpus $2 expert_slicing.py
    exit 0
elif [ $EXPERT_SLICING -eq 0 ]; then
    echo "Setting EXPERT_SLICING=0, EP_SIZE=$2"
    deepspeed --num_gpus $2 no_slicing.py
    exit 0
else
    echo "EXPERT_SLICING must be 0 or 1"
    exit 1
fi