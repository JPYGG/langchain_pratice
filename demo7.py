import datetime
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
from pydantic import BaseModel, Field

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

persist_dir = 'chroma_data_dir'  # 存放向量数据库的目录

# 一些YouTube的视频连接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# def load_youtube_with_ytdlp(url):
#     # 配置参数
#     ydl_opts = {
#         'skip_download': True,  # 不下载视频文件
#         'writesubtitles': True,  # 获取字幕
#         'writeautomaticsub': True,  # 如果没有手动字幕，获取自动生成的字幕
#         'subtitleslangs': ['zh-Hans', 'en'],  # 优先获取中文和英文
#         'quiet': True
#     }
#
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         # 1. 提取元数据
#         info = ydl.extract_info(url, download=False)
#
#         title = info.get('title', 'Unknown Title')
#         description = info.get('description', '')
#         view_count = info.get('view_count', 0)
#
#         # 2. 尝试获取文本内容 (yt-dlp 主要是获取元数据，若需全文脚本需配合 transcript 工具)
#         # 这里我们将元数据封装进 LangChain Document
#         content = f"Title: {title}\nDescription: {description}"
#
#         metadata = {
#             "source": url,
#             "title": title,
#             "view_count": view_count
#         }
#
#         return Document(page_content=content, metadata=metadata)
#
# docs = [] # document的数组
# for url in urls:
#     # 一个youtube的视频对应一个document
#     try:
#         # loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
#         # docs.extend(loader.load())
#         docs.extend(load_youtube_with_ytdlp(url))
#     except Exception as e:
#         print(f"解析失败 {url}: {e}")
#         # 可以在这里记录日志，继续处理下一个
#         continue
#
#
# print(len(docs))
# print(docs[0])
# # 给doc 添加额外的元数据: 视频发布的年份
# for doc in docs:
#     doc.metadata['publish_year'] = int(
#             datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
#
# print(docs[0].metadata)
# print(docs[0].page_content[:500]) # 第一个视频的字幕内容
#
# # 根据多个doc构建向量数据库
# splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
# split_doc = splitter.split_documents(docs)
#
# # 向量数据库持久化到磁盘
# vectorstore = Chroma.from_documents(split_doc, embedding=safe_embeddings, persist_directory=persist_dir)

# 加载磁盘中的向量数据库
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 测试向量数据库的相似检索
# result = vectorstore.similarity_search_with_score('how do I build a RAG agent')
# print(result[0])
# print(result[0][0].metadata['publish_year'])


system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# pydantic
class Search(BaseModel):
    """
    定义了一个数据模型
    """
    # 内容的相似性和发布年份
    query: str = Field(None, description='Similarity search query applied to video transcripts.')
    publish_year: Optional[int] = Field(None, description='Year video was published')


chain = {'question': RunnablePassthrough()} | prompt | model.with_structured_output(Search)

# resp1 = chain.invoke('how do I build a RAG agent?')
# print(resp1)
# resp2 = chain.invoke('videos on RAG published in 2023')
# print(resp2)


def retrieval(search: Search) -> list[Document]:
    _filter = None
    if search.publish_year:
        # 根据publish_year，存在得到一个检索条件
        # "$eq"是Chroma向量数据库的固定语法
        _filter = {'publish_year': {"$eq": search.publish_year}}

    return vectorstore.similarity_search(search.query, filter=_filter)

new_chain = chain | retrieval

# result = new_chain.invoke('videos on RAG published in 2023')
result = new_chain.invoke('RAG tutorial')
print([(doc.metadata['title'], doc.metadata['publish_year']) for doc in result])


