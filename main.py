import json
import os

from langchain.llms import *
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # 替换 HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.schema import HumanMessage

# 配置大模型API
os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = OpenAI(model_name="Qwen2.5-14B")

# 初始化嵌入模型
# llm_chat = ChatOpenAI(model_name="Qwen2.5-14B")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# 连接到Milvus数据库
db = Milvus(embedding_function=embedding, collection_name="arXiv",
            connection_args={"host": "10.58.0.2", "port": "19530"})


def translate_to_english(text):
    translation_prompt = f"请将以下中文翻译成英文: {text}"
    translation_result = llm_completion.invoke([HumanMessage(content=translation_prompt)])
    return translation_result.content


def query_abstract(question):
    attempts = 0
    while attempts < 3:
        # 翻译问题为英文
        english_question = translate_to_english(question)

        # 优化用户输入提示
        optimized_prompt = f"Please optimize the following query to better find relevant academic paper abstracts: {english_question}"
        optimized_query = llm_completion.invoke([HumanMessage(content=optimized_prompt)])

        # 获取响应内容
        optimized_query_content = optimized_query.content

        # 查询向量数据库
        docs = db.similarity_search(optimized_query_content, k=5)

        # 检查是否找到相关文档
        if docs:
            best_doc = docs[0]
            answer_prompt = f"Based on the following paper abstract, answer the question: {best_doc.page_content}\n\nQuestion: {english_question}"
            answer = llm_completion.invoke([HumanMessage(content=answer_prompt)])
            return {
                "answer": answer.content,
                "source": f"https://arxiv.org/abs/{best_doc.metadata['access_id']}"
            }
        else:
            # 如果没有找到相关文档，进行提示优化和查询迭代
            attempts += 1
            print(f"尝试 {attempts}/3：没有找到相关文档，正在重新优化查询。")

    return {"answer": "未能找到相关文档，请尝试其他问题。", "source": None}


def answer_questions(questions_file, output_file):
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    for item in questions:
        question = item['question']
        print(f"回答问题: {question}")
        result = query_abstract(question)
        item['answer'] = result['answer']
        item['source'] = result.get('source', '')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    questions_file = './questions.jsonl'
    output_file = 'answer.json'
    answer_questions(questions_file, output_file)
