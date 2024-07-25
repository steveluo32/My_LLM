from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vector_store(documents):
    vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
    return vectorstore
