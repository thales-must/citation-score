# DCRE

## python lib

python==3.13.9

```bash
pip install -r requirements.txt
```

## 创建环境变量

modify `.env.local` name to `.env`

1. 基础信息

paramater | description
--- | ---
PROJECT_NAME | 项目名称
SERVER_DIR | docker数据路径
SEED | 随机数，用于复现的一致性

2. 数据库

paramater | description
--- | ---
POSTGRES_USER | 用户名
POSTGRES_PASSWORD | 密码
POSTGRES_DB | 数据库名
POSTGRES_PORT | 端口

3. 大模型

paramater | description
--- | ---
LLM_KEY | 大模型key
LLM_URL | 大模型地址
LLM_MODEL | 模型

4. 论文公开库

paramater | description
--- | ---
API_KEY | 有key可以更小间隔请求

https://www.semanticscholar.org 官网可以申请

## 种子文章收集

需要连接数据库，以及使用*semantic scholar api*

文件为 `collect.ipynb`

首先按照类型和时间条件收集论文，并且存入数据库，然后分层取得种子文章

读取种子文章的引用文章，先记录上下文和摘要齐全的文章，先不入库

> 因为*semantic scholar api*没有提供按照需要的接口，所以需要保存数据之后，用数据库的功能取出需要的种子文章

## 作者同步

需要同步文章作者信息

文件为 `author.ipynb`

## 引用评分

按照种子文章读取所有引用文章，文章入库，文章关系入库，并且计算分数

文件为 `relevance.ipynb`

> 参考网站 https://sbert.net/

## 精选文章

筛选320条数据，8个分类，每个分类40条数据，40条数据按照cross-encoder的分数分4个阶段，每个阶段10篇

文件为  `subset.ipynb`

## 专家打标

竞选文章让专家打标，用来做后续的实验

专家打标的文件为 `expert_label.xlsx`

文件`label.ipynb`可以根据专家打标信息输出柱状图和热力图

## 语义模型对比

输出分布图，箱线图，spearman，mae，qwk以及热力图

文件`sbert.ipynb`

## 大模型选择

根据精选文章用大模型计算分数

文件`llm_model_select.ipynb`

输出分布图，箱线图，spearman，mae，qwk

文件`plot_llm_select.ipynb`

## 权重

调整权重计算spearman，mae，qwk

文件`llm_weight.ipynb`

## 上下文缺失惩罚

计算两篇文章摘要的分数

文件`script_abstract_score.py`

> 后台运行，支持续传

调整系数计算spearman，mae，qwk

文件`penalty.ipynb`

## 重新打分

重新打分，使用全文pdf，绘制图片以及表格

文件`rescore.ipynb`

## 作者专家调研

根据专家信息确定1跳和2跳，确定作者独立性分数

记录在`inform.xlsx`

商城分布图的文件为`inform.ipynb`

## 总分实验

全部数据读取总分和引用量的对比

文件`paper.ipynb`

可以查看总体柱状图，查看引用量和总分的差值，可以查看某个文章的分数过程