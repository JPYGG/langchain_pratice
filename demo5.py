import os
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""

# 调用大语言模型
#  创建模型
#

# 1. 加载数据: 一篇博客内容数据
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)

docs = loader.load()

# print(len(docs))
# print(docs)

# 2. 大文本切割
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = splitter.split_text(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

