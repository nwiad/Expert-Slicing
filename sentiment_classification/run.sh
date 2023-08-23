export EXPERT_SLICING=0
if [ $EXPERT_SLICING -eq 1 ]; then
    python expert_slicng.py
    exit 0
elif [ $EXPERT_SLICING -eq 0 ]; then
    python no_slicing.py
    exit 0
else
    echo "EXPERT_SLICING must be 0 or 1"
    exit 1
fi