## 运行准备：

- 用/revised_source/下的两个文件替换~/anaconda3/envs/[环境名]/lib/python3.x/site-packages/deepspeed/moe/下的同名文件，注意做好备份
- 安装requirements.txt中的依赖

## 运行方式：

- 在/expert_slicing/下执行`bash test_mlp.sh $1`对切片的MLP进行测试，`$1`代表切片数，是一个不超过机器可用GPU数目的正整数
- 在/expert_slicing/下执行`bash test_moe.sh $1 $2`对以MLP为专家的MoE模型进行测试，其中`$1`为0代表不进行专家切片，为1代表进行专家切片，`$2`代表切片数，是一个不超过机器可用GPU数目的正整数，需要保证整除expert_slicing.py中的EXPERTS_NUM
- 在/expert_slicing/下执行`bash loop.sh`进行批量对比实验，结果存储于res/results.txt中，使用plot.py生成图表

## 示例：

- `bash test_mlp.sh 2`表示将MLP进行二切片
- `bash test_moe.sh 1 4` 表示进行切片，使用4个GPU训练

## 测试结果：

在`batch_size=1, sequence_length=1, number_experts=4, TP_size=4`的条件下：

<img src="expert_slicing/pics_B1_SQ1/pics_W4_E4_TP4/line_W4_E4_TP4.png" alt="line_W4_E4_TP4" style="zoom: 67%;" /><img src="expert_slicing/pics_B1_SQ1/pics_W4_E4_TP4/sep_line_W4_E4_TP4.png" alt="sep_line_W4_E4_TP4" style="zoom:67%;" />

在`batch_size=8, sequence_length=1024, number_experts=4, TP_size=4`的条件下：

<img src="expert_slicing/pics_B8_SQ1024/pics_W4_E4_TP4/line_W4_E4_TP4.png" alt="line_W4_E4_TP4" style="zoom:67%;" /><img src="expert_slicing/pics_B8_SQ1024/pics_W4_E4_TP4/sep_line_W4_E4_TP4.png" alt="sep_line_W4_E4_TP4" style="zoom:67%;" />