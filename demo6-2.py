import os
import re
from operator import itemgetter

from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import chat_agent_executor
from langchain.agents import create_agent

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
# os.environ["LANGCHAIN_API_KEY"] = ""

class CleanOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # 移除 <thought> 及其内部所有内容
        cleaned_text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

# 调用大语言模型
#  创建模型
model = ChatOpenAI(
    model='gemma-4-31b-it',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

# sqlalchemy 初始化MySQL数据库的连接
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'sys'
USERNAME = 'root'
PASSWORD = '123456'
# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 创建工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 使用agent完整整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
system_message = SystemMessage(content=system_prompt)

# 创建代理
agent_executor = create_agent(model, tools, system_prompt=system_prompt)

resp = agent_executor.invoke({'messages': [HumanMessage(content='请问: 系统配置表有多少条数据?')]})
result = resp['messages']
print(result)
print(len(result))
# 最后一个才是真正的答案
print(result[len(result)-1])

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

