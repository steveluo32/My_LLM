from retrieve.retrieve_environment import *
from retrieve.model_initializer import initialize_model
from retrieve.data_loader import load_data
from retrieve.text_splitter import split_text
from retrieve.vector_store import create_vector_store
from retrieve.retriever import create_retriever
from retrieve.question_answering_chain import create_document_chain, execute_chain

def main_retrieve(question):
    # Initialize model
    model = initialize_model()

    # Load data
    data = load_data(clipped_path)

    # Split text into chunks
    all_splits = split_text(data)

    # Create vector store and retriever
    vectorstore = create_vector_store(all_splits)
    retriever = create_retriever(vectorstore)

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Create and execute question answering chain
    document_chain = create_document_chain(model)
    response = execute_chain(document_chain, question, docs)

    # Split and print the response
    retrieved_paths = response.split('\n')

    return retrieved_paths

if __name__ == "__main__":
    question = "Who is vincent jackson"
    retrieved_paths = main_retrieve(clipped_path, question)
    print(retrieved_paths)
