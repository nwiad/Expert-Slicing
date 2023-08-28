运行准备：
- 用/revised_source/下的两个文件替换~/anaconda3/envs/[环境名]/lib/python3.x/site-packages/deepspeed/moe/下的同名文件，注意做好备份
- 安装requirements.txt中的依赖

运行方式：
- 在/expert_slicing/下执行`sh test_mlp.sh $1`对切片的MLP进行测试，`$1`代表切片数，是一个不超过机器可用GPU数目的正整数
- 在/expert_slicing/下执行`sh test_moe.sh $1 $2`对以MLP为专家的MoE模型进行测试，其中`$1`为0代表不进行专家切片，为1代表进行专家切片，`$2`代表切片数，是一个不超过机器可用GPU数目的正整数，需要保证整除expert_slicing.py中的EXPERTS_NUM
- 在/expert_slicing/下执行`sh loop.sh`进行批量对比实验，HIDDEN_DIM依次取512 768 1024 1536 2048 2560 3072 4096 5140 6144 8192 10240 12288 16384，结果存储于results.txt中

示例：
- `sh test_mlp.sh 2`表示将MLP进行二切片
- `sh test_moe.sh 1 4` 表示进行切片，使用4个GPU训练