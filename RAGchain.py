from divisions.document_retriever import *
from divisions.answer_giver import *

class RAG_chain:
    def __init__(self):
        self.document_retriever = DocumentRetriever()
        self.answer_giver = AnswerGiver()

    def start_with_memory(self):
        print(start_msg())
        while True:
            question = input('Input: ')
            if question == "help" or question == "h":
                answer = help_msg()
            elif question == "exit" or question == "e":
                answer = exit_msg()
                print(answer)
                break
            else:
                retrieved_paths = self.document_retriever.retrieve_document(question)
                print("Retrieved Paths:")
                for path in retrieved_paths:
                    print(path)
                self.answer_giver.set_up(retrieved_paths)
                answer = self.answer_giver.get_answer_2(question)
            print(answer)

    def start_without_memory(self):
        print(start_msg())
        while True:
            question = input('Input: ')
            if question == "help" or question == "h":
                answer = help_msg()
            elif question == "exit" or question == "e":
                answer = exit_msg()
                print(answer)
                break
            else:
                retrieved_paths = self.document_retriever.retrieve_document(question)
                print("Retrieved Paths:")
                for path in retrieved_paths:
                    print(path)
                self.answer_giver.set_up(retrieved_paths)
                answer = self.answer_giver.get_answer_1(question)
            print(answer)

    def start_one_time(self, question):
        retrieved_paths = self.document_retriever.retrieve_document(question)
        print("Retrieved Paths:")
        for path in retrieved_paths:
            print(path)
        self.answer_giver.set_up(retrieved_paths)
        answer = self.answer_giver.get_answer_1(question)
        print(answer)

def start_msg():
    start_msg = """
    This is a chatbot for cis.unimelb.edu.au.
    You can input "help" or "h" to ask for help.
    Or you can input "exit" or "e" to exit the chatbot.
    Or you can input any query.\n
    """
    return start_msg

def help_msg():
    help_msg = """
    This is a chatbot for website cis.unimelb.edu.au.
    You can input any query related or unrelated to the website.
    The chatbot based on a pretrained model can give you answer based on the website.
    You can input "exit" or "e" to exit the chatbot.
    Or you can input any query.\n
    """
    return help_msg

def exit_msg():
    exit_msg = """
    You are exiting the chatbot..
    """
    return exit_msg