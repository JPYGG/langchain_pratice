import os
import re
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

# class Classification(BaseModel):
#     """
#         定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
#     """
#     # 文本的情感倾向，预期为字符串类型
#     sentiment: str = Field(description="文本的情感")
#
#     # 文本的攻击性，预期为1到10的整数
#     aggressiveness: int = Field(
#         description="描述文本的攻击性，数字越大表示越攻击性"
#     )
#
#     # 文本使用的语言，预期为字符串类型
#     language: str = Field(description="文本使用的语言")

# class Classification(BaseModel):
#     """
#         定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
#     """
#     # 文本的情感倾向，预期为字符串类型
#     sentiment: str = Field(..., enum=["happy", "neutral", "sad"], description="文本的情感")
#
#     # 文本的攻击性，预期为1到5的整数
#     aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5], description="描述文本的攻击性，数字越大表示越攻击性")
#
#     # 文本使用的语言，预期为字符串类型
#     language: str = Field(..., enum=["spanish", "english", "french", "中文", "italian"], description="文本使用的语言")
class Classification(BaseModel):
    # 使用 Literal 替代 Field 中的 enum 参数
    sentiment: Literal["happy", "neutral", "sad"] = Field(..., description="文本的情感")

    aggressiveness: Literal[1, 2, 3, 4, 5] = Field(..., description="描述文本的攻击性")

    language: Literal["spanish", "english", "french", "中文", "italian"] = Field(..., description="文本使用的语言")


# 创建一个用于提取信息的提示模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落中提取所需信息。
    只提取'Classification'类中提到的属性。
    段落：
    {input}
    """
)

chain = tagging_prompt | model.with_structured_output(Classification)

input_text = "中国人民大学的王教授：师德败坏，做出的事情实在让我生气！"
# input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
result: Classification = chain.invoke({'input': input_text})
print( result)

