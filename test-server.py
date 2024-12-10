import os
# from langchain_openai import *
# from langchain_core.messages import HumanMessage, SystemMessage
#
#
# os.environ["OPENAI_API_KEY"] = "None"
# os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
#
# llm_completion = ChatOpenAI(model_name="Qwen2.5-14B")
# print(llm_completion.invoke([HumanMessage("你好！")]))

from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")