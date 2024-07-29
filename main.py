from divisions.document_retriever import *
from divisions.answer_giver import *
from RAGchain import *

# TODO: Message Memory
# TODO: Token Limits

# if __name__ == "__main_":
#     # question = "Who is vincent jackson".lower()
#     # question = "Who is wei zhao".lower()
#     question = "hci projects".lower()
#     # question = "give me some students from cis".lower()
#     retrieved_paths = main_retrieve(question)
#     print("Retrieved Paths:")
#     for path in retrieved_paths:
#         print(path)
#     response = main_answer(question, retrieved_paths)
#     print(response)

# if __name__ == "__main__":
#     rag_chain = RAG_chain()
#     # question = "Who is vincent jackson"
#     question = "Who is wei zhao".lower()
#     # question = "Who is vinay kabadi".lower()
#     # question = "hci projects".lower()
#
#     rag_chain.start_one_time(question)

if __name__ == "__main__":
    rag_chain = RAG_chain()
    rag_chain.start_with_memory()