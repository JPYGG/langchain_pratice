import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import chat_agent_executor

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

# 调用大语言模型
# 1. 创建模型
model = ChatOpenAI(
    model='gemma-4-31b-it',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

# 没有任何代理的情况下
# result = model.invoke([HumanMessage(content='北京天气怎么样')])
# print(result)

# LangChain内置了一个工具, 可以轻松地使用Tavily搜索引擎作为工具
# max_results=2 只返回两个结果
search = TavilySearchResults(max_results=2)
# print(search.invoke('北京的天气怎么样?'))

# 让模型绑定工具
tools = [search]
# model_with_tools = model.bind_tools(tools)
#
# # 模型可以自动推理: 是否需要调用工具去完成用户的答案
# resp = model_with_tools.invoke([HumanMessage(content='中国的首都是哪个城市?')])
# print(f'Model_Result_Content: {resp.content}')
# print(f'Tools_Result_Content: {resp.tool_calls}')
#
# resp2 = model_with_tools.invoke([HumanMessage(content='北京天气怎么样?')])
# print(f'Model_Result_Content: {resp2.content}')
# print(f'Tools_Result_Content: {resp2.tool_calls}')

# 创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
resp = agent_executor.invoke({'messages': [HumanMessage(content='中国的首都是哪个城市?')]})
print(resp['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='北京天气怎么样?')]})
print(resp2['messages'])
