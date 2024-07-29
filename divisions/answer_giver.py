from config import *
from langchain_core.messages import HumanMessage
from langchain.docstore.document import Document
from components.model_initializer import chat_gpt
from components.utils import read_files
from components.text_splitter import recursive_character_splitter, semantic_splitter
from components.vectorstore_retriever import chroma_vectorstore, top_k_retriever, contextualcompression_retriever, \
    ensemble_retriever_2, historical_messages_retriever, ensemble_retriever_1
from components.question_answering_chain import execute_chain, create_document_chain_answer_giving

class AnswerGiver:
    def __init__(self):
        self.model = chat_gpt()
        self.data = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.chat_history = []

    def get_answer_1(self, question):
        docs = self.retriever.invoke(question)
        response = execute_chain(self.rag_chain, question, docs)
        return response

    def get_answer_2(self, question):
        docs = self.retriever.invoke({"chat_history": self.chat_history, "input": question})
        result = self.rag_chain.invoke({"context": docs, "messages": [HumanMessage(content=question)]})

        # Ensure result is a dictionary with the "answer" key
        if isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
        else:
            answer = result  # Fallback in case result is a string or unexpected format

        self.chat_history.extend([HumanMessage(content=question), HumanMessage(content=answer)])
        return answer

    def set_up(self, path_list):
        file_content = read_files(path_list)

        # Convert file content to Document object
        documents = [Document(page_content=file_content)]

        # Split text into chunks
        split_data = semantic_splitter(documents)

        # Set data
        self.data = split_data
        self.vectorstore = vector_store(self.data)
        self.retriever = create_retriever_2(self.data, self.model, self.vectorstore)
        self.rag_chain = create_rag_chain(self.model)

def vector_store(data):
    vectorstore = chroma_vectorstore(data)
    return vectorstore

def create_retriever(data, vectorstore, llm_model):
    # retriever = bm25_retriever(all_splits, 100)
    retriever = top_k_retriever(vectorstore, 100)
    retriever = ensemble_retriever_1(retriever, data, llm_model, 100)
    return retriever

def create_retriever_2(data, model, vectorstore):
    # retriever = bm25_retriever(all_splits, 100)
    retriever = top_k_retriever(vectorstore, 100)
    retriever = ensemble_retriever_1(retriever, data, model, 100)
    retriever = historical_messages_retriever(model, retriever)
    return retriever

def create_rag_chain(model):
    rag_chain = create_document_chain_answer_giving(model)
    return rag_chain

def main_answer(question, path_list):
    # Initialize model
    chat = chat_gpt()

    # Check if path_list is not empty and process accordingly
    if path_list:
        file_content = read_files(path_list)

        # Convert file content to Document object
        documents = [Document(page_content=file_content)]

        # Split text into chunks
        all_splits = semantic_splitter(documents)

        # Create vector store and retriever
        vectorstore = chroma_vectorstore(all_splits)
        retriever = top_k_retriever(vectorstore)
        retriever = contextualcompression_retriever(retriever, chat)

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)

        # Convert retrieved documents to messages
        messages = [HumanMessage(content=doc.page_content) for doc in retrieved_docs]

        # Add the question as the last message
        messages.append(HumanMessage(content=question))

        # Get response from the chat model
        response = chat.invoke(messages)
    else:
        messages = [HumanMessage(content=question)]
        response = chat.invoke(messages)

    return response.content


