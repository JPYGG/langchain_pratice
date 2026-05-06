import os

from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""

# 调用大语言模型
# 1. 创建模型
model = ChatOpenAI(
    model='gemma-4-31b-it',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

# 2. 准备prompt
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好, 请问你要去哪里? ')
]

result = model.invoke(msg)
# print(result)

# 简单的解析响应数据
# 3. 创建返回的数据解析器
parser = StrOutputParser()
# print(parser.invoke(result))

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译成{language}'),
    ('user', '{text}')
])


# 4. 得到链
chain = prompt_template | model | parser

# 5. 直接使用chain来调用
# print(chain.invoke(msg))
# print(chain.invoke({'language': 'English', 'text': '我下午还有一节课, 不能去打球了. '}))

# 把我们的程序部署成服务
# 创建fastAPI的应用
app = FastAPI(title='我的LangChain服务', version='V1.0', description='使用LangChain翻译任何语句的服务')

add_routes(
    app,
    chain,
    path='/chainDemo',
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)


