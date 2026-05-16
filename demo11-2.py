import os
import re

from langchain_classic.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


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
    timeout=60,       # 超过 60 秒强行断开并抛出异常，不再死卡
    max_retries=2     # 限制重试次数，防止无限循环
)
# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章
print("【进度】网页加载成功！")

# 第二种： Map-reduce
# 第一步：切割阶段
# 每一个小docs为1000个token
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # 针对中文和网页文本优化的降级策略
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

split_docs = text_splitter.split_documents(docs)


# # 第二步： map阶段
# map_template = """以下是一组文档(documents)
# "{docs}"
# 根据这个文档列表，请给出总结摘要:"""
# map_prompt = PromptTemplate.from_template(map_template)
#
# map_llm_chain = LLMChain(llm=model, prompt=map_prompt)
#
# # 第三步： reduce阶段：（combine和最终的reduce）
# reduce_template = """以下是一组总结摘要:
# {docs}
# 将这些内容提炼成一个最终的、统一的总结摘要:"""
# reduce_prompt = PromptTemplate.from_template(reduce_template)
# reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)
#
# '''
# reduce的思路:
# 如果map之后文档的累积token数超过了 4000个，那么我们将递归地将文档以<= 4000 个token的批次传递给我们的 StuffDocumentsChain 来创建批量摘要。
# 一旦这些批量摘要的累积大小小于 4000 个token，我们将它们全部传递给 StuffDocumentsChain 最后一次，以创建最终摘要。
# '''
#
# # 定义一个combine 的chain
# combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')
#
# reduce_chain = ReduceDocumentsChain(
#     # 这是最终调用的链
#     combine_documents_chain=combine_chain,
#     # 中间的汇总的链
#     collapse_documents_chain=combine_chain,
#     # 将文档分组的最大令牌数
#     token_max=4000
# )
#
#
# # 第四步：合并所有链
# map_reduce_chain = MapReduceDocumentsChain(
#     llm_chain=map_llm_chain,
#     reduce_documents_chain=reduce_chain,
#     document_variable_name='docs',
#     return_intermediate_steps=False,
# ).with_config(max_concurrency=2)
#
# # 第五步： 调用最终的链
# result = map_reduce_chain.invoke(split_docs)
# print( result['output_text'])
print(f"【进度】文档切分完成，生成了 {len(split_docs)} 个文本块。")

# 4. 用现代 LCEL 构造两条精简的链
# Map 阶段：处理单个文本块
map_prompt = ChatPromptTemplate.from_template("以下是一组文档:\n\"{docs}\"\n根据这个文档列表，请给出总结摘要:")
# 提示：如果你有自定义的 CleanOutputParser()，把下面的 StrOutputParser() 换成你的即可
map_chain = map_prompt | model | StrOutputParser()

# Reduce 阶段：合并所有摘要
reduce_prompt = ChatPromptTemplate.from_template("以下是一组总结摘要:\n{docs}\n将这些内容提炼成一个最终的、统一的总结摘要:")
reduce_chain = reduce_prompt | model | StrOutputParser()

# 5. 执行 Map 阶段（LCEL 的 batch 能够完美锁定并发数）
print(f"【进度】开始并行执行 Map 阶段。总共 {len(split_docs)} 个块，严格限制同时只发 2 个请求...")
map_inputs = [{"docs": doc.page_content} for doc in split_docs]

# 通过 config={"max_concurrency": 2} 强制让请求排队，一次只发2个，温柔对待 NVIDIA API
map_results = map_chain.batch(map_inputs, config={"max_concurrency": 2})
print("【进度】所有分块的 Map 摘要均已生成完毕！")

# 6. 执行 Reduce 阶段（合并并提炼最终结果）
print("【进度】开始执行最终的 Reduce 提炼...")
collapsed_docs = "\n\n".join(map_results)
final_summary = reduce_chain.invoke({"docs": collapsed_docs})

print("\n================ 最终总结摘要 ================")
print(final_summary)
