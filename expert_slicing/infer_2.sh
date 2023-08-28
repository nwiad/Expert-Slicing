export EXPERT_SLICING=$1
export EP_SIZE=$2
if [ $# -ne 2 ]; then
    echo "Error: 2 arguments ([EXPERT_SLICING] [PARALLEL_SIZE]) are needed"
    exit 1
fi
if [ $EXPERT_SLICING -eq 1 ]; then
    if [ $2 -le 1 ]; then
        echo "Error: TP_SIZE should be greater than 1 when doing expert slicing"
        exit 1
    fi
    echo "Setting EXPERT_SLICING=1, EP_SIZE=$2, TP_SIZE=$2"
    export TP_SIZE=$2
    deepspeed --num_gpus $2 infer_2.py
    exit 0
elif [ $EXPERT_SLICING -eq 0 ]; then
    echo "Setting EXPERT_SLICING=0, EP_SIZE=$2, TP_SIZE=1"
    export TP_SIZE=1
    deepspeed --num_gpus $2 infer_2.py
    exit 0
else
    echo "EXPERT_SLICING must be 0 or 1"
    exit 1
fi