import os
import re

from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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
    model='openai/gpt-oss-120b',
    temperature=0,
    base_url='https://integrate.api.nvidia.com/v1',
)
# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章

# 第一种： Stuff

# Stuff的第一种写法
chain = load_summarize_chain(model, chain_type='stuff')

# Stuff的第二种写法
# 定义提示
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=model, prompt=prompt)

stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')

result = chain.invoke(docs)
print( result['output_text'])



