运行准备：

- 用revised_source/下的两个文件替换~/anaconda3/envs/[环境名]/lib/python3.x/site-packages/deepspeed/moe/下的同名文件，注意做好备份
- 安装expert_slicing/requirements.txt中的依赖

运行方式：在expert_slicing/下执行`sh run.sh $1 $2`，其中`$1`为0代表不进行切片，为1代表进行切片，`$2`是一个不超过机器可用GPU数目的正整数，需要保证整除expert_slicing.py中的EXPERTS_NUM

示例：`sh run.sh 1 4` 表示进行切片，使用4个GPU训练