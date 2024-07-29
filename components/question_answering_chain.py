from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

document_retrieval_template = """
        Below are the file paths without punctuation. 
        Retrieve all the file paths that may contain the answer to the user's question. 
        Provide as many paths as possible.
        <web info>
        {context}
        </web info>
        """

answer_giving_template = """
        <web info>
        {context}
        </web info>
        """

contextualize_q_system_prompt_template = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is.
        """

document_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            document_retrieval_template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

answer_giving_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            answer_giving_template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def create_document_chain_document_retrieval(model):
    document_chain = create_stuff_documents_chain(model, document_retrieval_prompt)
    return document_chain


def create_document_chain_answer_giving(model):
    document_chain = create_stuff_documents_chain(model, answer_giving_prompt)
    return document_chain


def create_retrieval_chain(retriever, question_answer_chain):
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    return retrieval_chain

def execute_chain(document_chain, question, context):
    messages = [
        HumanMessage(content=question),
    ]
    response = document_chain.invoke({"messages": messages, "context": context})
    return response
