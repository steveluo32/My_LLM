# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
#
# template = """Question: {question}
#
# Answer: Let's think step by step."""
#
# prompt = ChatPromptTemplate.from_template(template)
#
# model = OllamaLLM(model="llama3.1")
#
# chain = prompt | model
#
# response = chain.invoke({"question": "数字8.11和8.9哪个大"})
#
# print(response)