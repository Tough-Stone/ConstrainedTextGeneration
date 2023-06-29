# 任务介绍
约束性中文文本生成，根据关键词生成军事报道文本，要求关键词在文本中全部出现且出现顺序不变。
<br>数据来源：https://www.datafountain.cn/competitions/633

# 项目结构
* checkpoints：存放模型文件
  * bart-large-chinese：中文bart预训练模型
  * large-finetune-model：经过赛题数据finetune后的模型
  * gpt2通用中文模型：排序模型
* dataset：存放数据集文件
  * original：原始数据集
  * preprocess：预处理后的数据集（用于获取合成数据,包括分词数量、词token、tf-idf分数）
  * synthetic：最终获取的合成数据（训练数据）
* log：存放日志文件
* output：存放模型的推理输出结果
* src：源码
  * transformers：搬运第三方库transformers中需要导入的文件（添加了BartForTextInfill）
  * utils：一些工具
    * Log.py：日志生成代码
    * RankModel：排序模型（用于从多个decoder输出选取最优）
  * DataProcess.py：将初始数据集文件制作为合成数据集
  * Train.py：训练代码
  * Log.py：日志记录代码
  * submit.py：将输出结果转化json格式
***

# 使用方法
1. 安装依赖包
   * python >= 3.6
   * torch >= 1.4.0
   * pympler >= 0.8
   * requests
   * filelock
   * tqdm
   * tokenizers
   * regex
   * sentences
   * six
   * sacremoses
   * 如果训练完成的模型已经放入checkpoints文件夹，可直接跳到第5步做推理
2. 下载中文bart预训练模型([base](https://huggingface.co/fnlp/bart-base-chinese) / [large](https://huggingface.co/fnlp/bart-large-chinese))和[中文GPT2预训练模型](https://github.com/Morizeyao/GPT2-Chinese)，放入checkpoints文件夹
3. 处理数据集文件:`python DataProcess.py`，或下载可直接用于训练的[数据文件](https://huggingface.co/datasets/ToughStone/ConstrainedTextGeneration)
4. 训练模型：`python Train.py`，或下载训练好的[模型](https://huggingface.co/ToughStone/large-finetune-model)，放入checkpoints文件夹
5. 利用训练好的模型推理测试集结果：`python Inference.py`
***

# 效果展示
* 例1：
<br>Key works:	前苏联，部署，核武器，核反应堆
<br>Generated sentence:	美日在前苏联瓦解体时部署的核武器,其中包括核反应堆｡
* 例2：
<br>Key works:	雷达，导弹，护卫舰
<br>Generated sentence:	据美军阿海航母近距雷达拍的照,在这艘巨大战船靠前进港巡逻中还发现有一艘导弹护卫舰上的武器最先被曝光｡


# 参考
## 原文
> [【EMNLP2021】Parallel Refinements for Lexically Constrained Text Generation with BART](https://arxiv.org/abs/2109.12487)
## 改动
1. 合成数据
   * 先对训练文本分词，从分词后的结果随机抽取关键词作为训练数据。
   * 控制台参数：`max_insert_label`，这里仅取1。
   * 控制台参数：`insert_mode`取值范围[0, 1, 2, 3, 4]，这里仅取0。
2. 训练
   * 控制台参数：`insert_mode`取值范围[0, 1, 2, 3, 4]，这里仅取0。
   * 控制台参数：`generate_mode`取值范围[0, 1, 2]，这里仅取0。 
   * 控制台参数：`random_init`取值范围[0, 1]，这里仅取0。
   * 控制台参数：去掉了`local_rank=-1`默认值。
3. 推理
   * 控制台参数: `refinement_steps`10->100。
   * 控制台参数: `max_refinement_steps`30->300。
   * 控制台参数: `max_len`40->400。
***