import os
import re
from operator import itemgetter

from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI


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

# 测试连接是否成功
# print(db.get_usable_table_names())
# print(db.run('select * from sys_config;'))

# 直接使用大模型和数据库整合, 只能根据你的问题生成SQL
# 初始化生成SQL的chain
test_chain = create_sql_query_chain(model, db)
test_chain = test_chain | CleanOutputParser()

# resp = test_chain.invoke({'question': '请问系统配置表中有多少条数据?'})
# print(resp)

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)
# 创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 1、生成SQL，2、执行SQL
# 2、模板
chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql_tool)
         | answer_prompt
         | model
         | CleanOutputParser()
         )
rep = chain.invoke(input={'question': '请问系统配置表中有多少条数据?'})
print(rep)



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

