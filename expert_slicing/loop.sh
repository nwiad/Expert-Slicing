sizes=(512 768 1024 1536 2048 2560 3072 4096 5140 6144 8192 10240 12288 16384)

# echo "Looping for W2 E2 TP2"
# for i in "${sizes[@]}"
# do
#     > bin/sliced.txt
#     > bin/unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer.sh 1 2 2 2
#     sh infer.sh 0 2 2 2
#     python cal.py
# done

echo "Looping for W2 E4 TP2"
for i in "${sizes[@]}"
do
    > bin/sliced.txt
    > bin/unsliced.txt
    export HIDDEN_DIM=$i
    sh infer.sh 1 2 4 2
    sh infer.sh 0 2 4 2
    python cal.py
done

echo "Looping for W4 E4 TP2"
for i in "${sizes[@]}"
do
    > bin/sliced.txt
    > bin/unsliced.txt
    export HIDDEN_DIM=$i
    sh infer.sh 1 4 4 2
    sh infer.sh 0 4 4 2
    python cal.py
done

echo "Looping for W4 E4 TP4"
for i in "${sizes[@]}"
do
    > bin/sliced.txt
    > bin/unsliced.txt
    export HIDDEN_DIM=$i
    sh infer.sh 1 4 4 4
    sh infer.sh 0 4 4 4
    python cal.py
done

# echo "Looping for W4 E4 TP4 step100"
# for i in $(seq $1 100 $2)
# do
#     > bin/sliced.txt
#     > bin/unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer.sh 1 4 4 4
#     sh infer.sh 0 4 4 4
#     python cal.py
# done