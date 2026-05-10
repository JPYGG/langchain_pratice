import os
import bs4
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = ""

# 调用大语言模型
#  创建模型
model = ChatOpenAI(
    model='gemma-4-31b-it',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
)

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

splits = splitter.split_documents(docs)

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


# 2. 使用包装后的模型
safe_embeddings = SafeGoogleEmbeddings(embeddings)

# 2. 存储
vectorstore = Chroma.from_documents(documents=splits, embedding=safe_embeddings)

# 3. 检索器
retriever = vectorstore.as_retriever()


# 整合
# 创建一个问题的模板
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", system_prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(model, prompt)

chain2 = create_retrieval_chain(retriever, chain1)

resp = chain2.invoke({'input': "What is Task Decomposition?"})
print(resp['answer'])



