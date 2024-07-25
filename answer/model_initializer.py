from langchain_openai import ChatOpenAI

def initialize_model(model_name="gpt-4", temperature=0.1):
    chat = ChatOpenAI(model=model_name, temperature=temperature)
    return chat
