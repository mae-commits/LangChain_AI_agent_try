import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# .env ファイルから環境変数を読み込む
load_dotenv()

# 必要な環境変数を取得
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "langchain_try"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# ドキュメントを読み込む関数
def document_loader(file_path, persist_directory="db_storage"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 永続化ディレクトリが存在するか確認
    if os.path.exists(persist_directory):
        print(f"既存のデータベースをロードしています: {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("新しいデータベースを作成します")
        # WebBaseLoader でドキュメントを読み込む
        loader = WebBaseLoader(file_path)
        documents = loader.load()

        # Chroma の永続化設定
        db = Chroma.from_documents(
            documents, embeddings, persist_directory=persist_directory
        )
    return db

# メッセージを生成する関数
def create_messages(db, model):
    prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問: {question}
    ''')
    retriever = db.as_retriever()

    chain = {
        "question": RunnablePassthrough(),
        "context": retriever,
    } | prompt | model | StrOutputParser()
    
    return chain.invoke("合格方法を教えて")

st.title("資格試験に向けた勉強をしよう！")

# file_path = "https://zenn.dev/pe/articles/9eee21eae6f2a3"

user_input = st.text_input("参照したい資格試験の体験記の URL を入力してください。")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

if user_input:
    db = document_loader(file_path=user_input)
    with st.spinner("合格方法を作成中..."):
        output = create_messages(db, model)
        st.write(output)

# st.write(f"{docs[0].page_content}")

# if user_input:
#   with st.spinner("レシピを作成中..."):
#     output = create_messages({"dish": user_input}, model)
#     st.write(output)