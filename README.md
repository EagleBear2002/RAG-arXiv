# RAG-arVix

## 如何运行

因为网络原因，运行时无法自动下载
sentence-transformers，需要手动下载：[sentence-transformers](https://hf-mirror.com/sentence-transformers/all-MiniLM-L12-v2/tree/main)
整个目录。对于根目录下 `pytorch_model.bin` 和 `model.safetensors` 两个文件，需要手动下载。

安装正确版本的依赖：

```
pip install openai==0.28
pip install pymilvus==2.2.6
```

其他依赖直接安装最新版即可。

## 项目功能

构建一个针对 arXiv 的知识问答系统：

- 给定一个入口，用户可以输入提问
- 不要求要求构建 GUI 界面
- 用户通过对话进行交互

系统寻找与问题相关的论文 abstract：

- 使用用户的请求对向量数据库进行请求
- 寻找与问题最为相关的 abstract
- 系统根据问题和论文 abstract 回答用户问题，并给出解答问题的信息来源

## 进阶功能：提示优化

- 用户给出的问题或陈述不一定能够匹配向量数据库的查询
- 使用大模型对用户的输入进行润色，提高找到对应文档的概率
- 思路提示（解决思路不唯一，提示仅作为可能的思路示例）
    - 观察不同输入后向量数据库找到对应文档的概率
    - 总结适用于查询的语句
    - 构建提示（prompt）实现对用户输入的润色
- 查询迭代
    - 单次的查询可能无法寻找到用户所期望的答案
    - 需要通过多轮的搜索和尝试才能获得较为准确的答案
    - 思路提示：
        - 如何将用户的需求拆解，变成可以拆解的逻辑步骤
        - 如何判断已经获得准确的答案并停止迭代
        - 如何再思路偏移后进行修正

## 资源简介

向量数据库可按需自行部署，大模型可选择自有 api，下列内容为能完成任务的所需资源。

arxiv 数据集可以从这里获取 https://www.kaggle.com/datasets/Cornell-University/arxiv

### 大模型（Qwen2.5-14B）

Qwen2.5-14B 模型已接入 LangChain Openai API，调用示例如下。注意，此处修改了 Moodle 中提供的代码段！

```python
from langchain.llms import OpenAI, OpenAIChat
import os

os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = ChatOpenAI(model_name="Qwen2.5-14B")
```

openai 包使用特定版本，避免与 langchain 不兼容 `pip install openai==0.28`

langchain 是用于方便整合大模型和向量数据库，也可以选择不使用。

### 嵌入模型（sentence-transformers/all-MiniLM-L12-v2）

嵌入模型使用 huggingface 中的 all-MiniLM-L12-v2 模型。

```python
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="./all-MiniLM-L12-v2")
```

### 向量数据库

arXiv 数据存在 Milvus 中

```python
from langchain.vectorstores import Milvus

db = Milvus(embedding_function=embedding, collection_name="arXiv",
            connection_args={"host": "10.58.0.2", "port": "19530"})
```

由于向量数据库与 SDK 存在强绑定关系，安装 milvus 包时请检查版本： `pip install pymilvus==2.2.6`

#### 数据项解释

- `vector`： 论文 abstract 的向量化表示
- `access_id`：论文的唯一 id
- `https://arxiv.org/abs/{access_id}` 论文的详情页
- `https://arxiv.org/pdf/{access_id}` 论文的 pdf 地址
- `authors`：论文的作者
- `title`：论文的题目
- `comments`：论文的评论，一般为作者的补充信息
- `journal_ref`：论文的发布信息
- `doi`：电子发行 doi
- `text`：论文的 abstract (为了兼容 langchain 必须命名为 text)
- `categories`：论文的分类

### LangChain

- langchain 官方文档 https://python.langchain.com/
- langchain 官方课程 https://learn.deeplearning.ai/langchain

## 提交内容

代码实现
预置题目
预置由 json 文件构成，包含 10 个问题(question 项目)
使用算法回答其中问题，答案存在 answer 项内
该文件存储为 answer.json，单独提交
完成过程中的技术问题可以联系:

