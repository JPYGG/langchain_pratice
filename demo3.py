import os

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""

# 调用大语言模型
# 1. 创建模型
model = ChatOpenAI(
    model='gemma-4-31b-it',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# 实例化 Google 原生的嵌入模型
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
)

# 1. 确保使用 valid_docs
valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
# # 2. 实例化嵌入模型 (建议使用 text-embedding-004)
# # embeddings_model = GoogleGenerativeAIEmbeddings(
# #     model="models/gemini-embedding-2",
# #     google_api_key="AIzaSyAITspP7M_WMMwdbSX-WabDnkdTLRsDoLg"
# # )
#
# # 3. [调试用] 打印数量对比，确认 API 是否偷懒了
# all_texts = [d.page_content for d in valid_docs]
# vectors = embeddings.embed_documents(all_texts)
# print(f"发送文本数: {len(all_texts)}, 返回向量数: {len(vectors)}")
#
# # 4. 如果数量一致，再存入 Chroma
# if len(all_texts) == len(vectors):
#     # 注意这里传入的是 valid_docs，而不是 documents
#     vector_store = Chroma.from_documents(
#         documents=valid_docs,
#         embedding=embeddings
#     )
#     print("向量库创建成功！")
# else:
#     print("错误：API 返回的向量数量与文档数量不符，请检查文档内容。")
#
#
# # 3. 如果测试成功，再存入向量库
# # 实例化一个向量空间
# vector_store = Chroma.from_documents(documents, embedding=embeddings)

# 1. 定义一个“安全包装类”，修复 Google API 的批处理 Bug
class SafeGoogleEmbeddings:
    def __init__(self, genai_embeddings):
        self.model = genai_embeddings

    def embed_documents(self, texts):
        # 核心修复：内部改为逐条调用，确保 5 个进 5 个出
        return [self.model.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.model.embed_query(text)


# 2. 使用包装后的模型
safe_embeddings = SafeGoogleEmbeddings(embeddings)

# 3. 现在你可以回到最舒服的写法了，逻辑完全兼容
vector_store = Chroma.from_documents(
    documents=valid_docs,
    embedding=safe_embeddings
)

# 检查 Chroma 集合中的文档总数
print(f"向量库中的文档总数: {vector_store._collection.count()}")

# 相似度的查询: 返回相似的分数, 分数越低相似度越高
# print(vector_store.similarity_search_with_score('咖啡猫'))
# 将 k 设置为 5，确保检索范围覆盖所有文档
# query_results = vector_store.similarity_search_with_score('咖啡猫', k=5)

# 打印结果
# for doc, score in query_results:
#     print(f"得分: {score:.4f} | 内容: {doc.page_content}")

# 检索器 bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# print(retriever.batch(['咖啡猫', '鲨鱼']))

# 提示模版
message = """
使用提供的上下文仅回答这个问题.
{question}
上下文: 
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough 允许我们将用户的问题之后再传递给prompt和model
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model

resp = chain.invoke('请介绍一下猫?')

print(resp.content)
