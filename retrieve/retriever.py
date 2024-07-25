def create_retriever(vectorstore, k=5):
    retriever = vectorstore.as_retriever(k=k)
    return retriever
