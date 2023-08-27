if [ $# -eq 0 ]
    then
    echo "No arguments supplied, setting TP_SIZE to 2"
    export TP_SIZE=2
elif [ $# -eq 1 ]
    then
    echo "Setting TP_SIZE to $1"
    export TP_SIZE=$1
else
    echo "Error: Only 1 argument is needed"
    exit 1
fi
deepspeed --num_gpus $1 test_mlp.py