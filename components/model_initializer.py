from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from langchain_ollama.llms import OllamaLLM


def chat_gpt(model_name="gpt-4o", temperature=0):
    model = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=3000)
    return model


def gemini(model_name="gemini-1.5-pro", temperature=0):
    model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return model


def vertex(model_name="gemini-1.5-pro", temperature=0):
    model = VertexAI(model_name=model_name, temperature=temperature)
    return model


def llama(model_name="llama3.1"):
    model = OllamaLLM(model=model_name)
    return model


if __name__ == "__main__":
    # llama_3_1()
    llama()
