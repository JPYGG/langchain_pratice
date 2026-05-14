import os
import re

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
    temperature=0.8,
    base_url='https://integrate.api.nvidia.com/v1',
)

# 生成一些结构化的数据： 5个步骤
# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int = Field(description="患者ID，整数类型")
    patient_name: str = Field(description="患者姓名，字符串类型")
    diagnosis_code: str = Field(description="诊断代码，字符串类型")
    procedure_code: str = Field(description="程序代码，字符串类型")
    total_charge: float = Field(description="总费用，浮点数类型")
    insurance_claim_amount: float = Field(description="保险索赔金额，浮点数类型")

# 如果你想一次性生成多条数据，可以定义一个容器模型
class MedicalBillingList(BaseModel):
    items: list[MedicalBilling] = Field(description="医疗账单列表")

# 2、 提供一些样例数据，给AI
# examples = [
#     {
#         "example": "Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
#     },
#     {
#         "example": "Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
#     },
#     {
#         "example": "Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"
#     },
# ]
#
# # 3. 创建一个提示模版, 用来指导AI生成符合规定的数据
# openai_template = PromptTemplate(input_variables=['example'], template="{example}")
#
# prompt_template = FewShotPromptTemplate(
#     prefix=SYNTHETIC_FEW_SHOT_PREFIX,
#     suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
#     examples=examples,
#     example_prompt=openai_template,
#     input_variables=['subject', 'extra']
# )
#
# # 4. 创建一个结构化数据的生成器
# generator = create_openai_data_generator(
#     output_schema=MedicalBilling, #指定输出数据的格式
#     llm=model,
#     prompt=prompt_template
# )
#
# # 5. 调用生成器
# result = generator.generate(
#     subject='医疗账单', # 指定生成数据的主题
#     extra='名字可以是随机的, 最好使用比较生僻的人名. ', # 额外的一些指导信息
#     runs=10, # 指定生成数据的数量
# )
# print(result)

# 3. 绑定结构化输出 (核心逻辑)
# 这里我们绑定 MedicalBillingList 以支持 runs=10 这种批量生成的需求
structured_llm = model.with_structured_output(MedicalBillingList)

# 4. 创建提示模板
# 这种方式下，我们直接在 Prompt 里描述规则和示例
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个医疗数据生成器。根据用户提供的主题和指导信息，生成符合格式的随机合成数据。"),
    ("human", """
生成主题: {subject}
指导信息: {extra}
生成数量: {count}

示例参考:
- Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350
- Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120
""")
])

# 5. 组合 Chain 并运行
chain = prompt | structured_llm

result = chain.invoke({
    "subject": "医疗账单",
    "extra": "医疗总费用呈现正态分布，最小的总费用为1000",
    "count": 10
})

# 6. 打印结果
for i, item in enumerate(result.items):
    print(f"数据 {i+1}: {item}")



