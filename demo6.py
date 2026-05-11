import os
import bs4
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""

# 调用大语言模型
#  创建模型
# model = ChatOpenAI(
#     model='gemma-4-31b-it',
#     base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
# )

# sqlalchemy 初始化MySQL数据库的连接
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'sys'
USERNAME = 'root'
PASSWORD = '123456'
# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

print(db.get_usable_table_names())
print(db.run('select * from sys_config;'))






# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-2",
# )
# class SafeGoogleEmbeddings:
#     def __init__(self, genai_embeddings):
#         self.model = genai_embeddings
#
#     def embed_documents(self, texts):
#         # 核心修复：内部改为逐条调用，确保 5 个进 5 个出
#         return [self.model.embed_query(text) for text in texts]
#
#     def embed_query(self, text):
#         return self.model.embed_query(text)
#
#
# # 2. 使用包装后的模型
# safe_embeddings = SafeGoogleEmbeddings(embeddings)

