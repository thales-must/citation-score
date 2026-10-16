# DCRE

## python lib

python==3.13.9

```bash
pip install -r requirements.txt
```

## 创建环境变量

modify `.env.local` name to `.env`

1. 基础信息

Parameter | description
--- | ---
PROJECT_NAME | 项目名称
SERVER_DIR | docker数据路径
SEED | 随机数，用于复现的一致性

2. 数据库

Parameter | description
--- | ---
POSTGRES_USER | 用户名
POSTGRES_PASSWORD | 密码
POSTGRES_DB | 数据库名
POSTGRES_HOST | 数据库地址
POSTGRES_PORT | 端口

3. 大模型

Parameter | description
--- | ---
LLM_KEY | 大模型key
LLM_URL | 大模型地址
LLM_MODEL | 模型

4. 论文公开库

Parameter | description
--- | ---
API_KEY | 有key可以更小间隔请求

https://www.semanticscholar.org 官网可以申请

## 种子文章收集

需要连接数据库，以及使用*semantic scholar api*

文件为 `collect.ipynb`

收集2024年发布的所有论文

> 因为*semantic scholar api*没有提供按照需要的接口，所以需要保存数据之后，用数据库的功能取出需要的种子文章

筛选320条数据，8个分类，每个分类40条数据，40条数据按照cross-encoder的分数分4个阶段，每个阶段10篇

## 作者同步

需要同步文章作者信息，并且把作者之前发布的论文导入

文件为 `author.ipynb`

> 导入作者信息时候也会导入作者发布的论文，论文很多是2024年之前发布的，为了查看作者之前是否有过合作

## 种子文章引用覆盖分析

文件为 `data_sampling.ipynb`

## 引用评分

给所有包含context的数据做关联性分析

文件为 `relevance.ipynb`

> 参考网站 https://sbert.net/

## 精选文章

筛选320条数据，8个分类，每个分类40条数据，40条数据按照cross-encoder的分数分4个阶段，每个阶段10篇

文件为 `subset.ipynb`

## 专家打标

竞选文章让专家打标，用来做后续的实验

专家打标的文件为 `expert_label.xlsx`

文件`label.ipynb`可以根据专家打标信息输出柱状图和热力图

## 根据专家打标做验证

给所有的专家打标过的数据完成大模型评分

文件``

- 验证向量模型
- 大模型选型
- 权重确定
- 引用上下文缺失的惩罚系数

文件`expert_evaluation.ipynb`

## 重新打分

重新打分，使用全文pdf，绘制图片以及表格

文件`dao_rescore.ipynb`

> 要在获得特别样本时候，手动找到相关的文章pdf

## 独立性

根据专家信息确定1跳和2跳，确定作者独立性分数

记录在`survey.xlsx`

确定独立性系数之后可以全量运行

``` bash
python .\score_all_citations.py --write --max-llm-calls 10000 
```

> 会在数据库中更新数据，每一条数据都记录cross-encoder分数llm分数，以及context缺失时候abstract的分数，还有作者独立性分数

查看所有作者独立性相关参数

文件为`author_independence.ipynb`

## 总分实验

全部数据读取总分和引用量的对比

文件`count_score.ipynb`

可以查看总体柱状图，查看引用量和总分的差值，可以查看某个文章的分数过程
