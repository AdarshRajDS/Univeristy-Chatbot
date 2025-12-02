from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List

from langsmith import Client as LangSmithClient

hub = LangSmithClient()



from langchain_chroma import Chroma
import os
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from .consts import INDEX_NAME

# Initialize Hub


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # --- Using Groq LLM ---
    chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ["GROQ_API_KEY"]
)


    # --- Pull prompts from modern Hub ---
    rephrase_prompt = hub.pull_prompt("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull_prompt("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    return qa.invoke({"input": query, "chat_history": chat_history})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    chat = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        verbose=True
    )

    rephrase_prompt = hub.get("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.get("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = (
        RunnablePassthrough
        .assign(context=retrieve_docs_chain)
        .assign(answer=rag_chain)
    )

    return chain.invoke({"input": query, "chat_history": chat_history})
