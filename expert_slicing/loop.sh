for i in $(seq $1 500 $2)
do
    > sliced.txt
    > unsliced.txt
    export HIDDEN_DIM=$i
    sh infer.sh 1 4
    sh infer.sh 0 4
    python cal.py
done