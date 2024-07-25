from retrieve.main_retrieve import *
from answer.main_answer import *

if __name__ == "__main__":
    question = "Who is vincent jackson".lower()
    # question = "Who is wei zhao".lower()
    # question = "hci projects".lower()
    retrieved_paths = main_retrieve(question)
    processed_retrieved_paths = process_retrieved_paths(retrieved_paths)
    for path in processed_retrieved_paths:
        print(path)
    response = main_answer(question, processed_retrieved_paths)
    print(response)