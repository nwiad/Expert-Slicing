# for i in $(seq $1 100 $2)
# do
#     > sliced.txt
#     > unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer.sh 1 2
#     sh infer.sh 0 2
#     python cal.py
# done
sizes=(512 768 1024 1536 2048 2560 3072 4096 5140 6144 8192 10240 12288 16384)
# W4 E4 TP2
# for i in "${sizes[@]}"
# do
#     > sliced.txt
#     > unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer.sh 1 2
#     sh infer.sh 0 2
#     python cal.py
# done
# W2 E2 TP2
# for i in "${sizes[@]}"
# do
#     > sliced.txt
#     > unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer_2.sh 1 2
#     sh infer_2.sh 0 2
#     python cal.py
# done
# W4 E4 TP4
# for i in "${sizes[@]}"
# do
#     > sliced.txt
#     > unsliced.txt
#     export HIDDEN_DIM=$i
#     sh infer.sh 1 4
#     sh infer.sh 0 4
#     python cal.py
# done