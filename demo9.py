import os
import re
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

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
    temperature=0.8,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
)
class SafeGoogleEmbeddings:
    def __init__(self, genai_embeddings):
        self.model = genai_embeddings

    def embed_documents(self, texts):
        # 核心修复：内部改为逐条调用，确保 5 个进 5 个出
        return [self.model.embed_query(text) for text in texts]

    def embed_query(self, text):
        return self.model.embed_query(text)

safe_embeddings = SafeGoogleEmbeddings(embeddings)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
result = chain(  # 给于一些关键词， 随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result['text'])





