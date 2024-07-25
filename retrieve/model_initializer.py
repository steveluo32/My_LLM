from langchain_openai import ChatOpenAI

def initialize_model(model_name="gpt-4o", temperature=0.1):
    model = ChatOpenAI(model=model_name, temperature=temperature)
    return model
