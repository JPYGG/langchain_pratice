import os
import re
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""

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

# pydantic: 处理数据, 验证数据, 定义数据的格式, 虚拟化和反虚拟化, 类型转换等等

# 定义一个数据
class Person(BaseModel):
    """
    关于一个人的数据模型
    """
    name: Optional[str] = Field(default=None, description='表示人的名字')

    hair_color: Optional[str] = Field(
        default=None, description="如果知道的话，这个人的头发颜色"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="以米为单位测量的高度"
    )

class ManyPerson(BaseModel):
    """
    数据模型类： 代表多个人
    """
    people: list[Person]


# 定义自定义提示以提供指令和任何其他上下文。
# 1) 你可以在提示模板中添加示例以提高提取质量
# 2) 引入额外的参数以考虑上下文（例如，包括有关提取文本的文档的元数据。）
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的提取算法。只从未结构化文本中提取相关信息。如果你不知道要提取的属性的值，返回该属性的值为null。",
        ),
        # 请参阅有关如何使用参考记录消息历史的案例
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)


# with_structured_output 模型的输出是一个结构化的数据
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson)

# text = '马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右，'

# text = "马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右。走在她旁边的是她的男朋友，叫：刘海；比她高10厘米。"
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
resp = chain.invoke(text)
print(resp)


