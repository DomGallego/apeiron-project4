import os
import shutil
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, GitLoader
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain import hub




from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain import VectorDBQA, OpenAI
import random
import qdrant_client
from qdrant_client import models, QdrantClient
from transformers import AutoTokenizer
from langchain.chains.question_answering import load_qa_chain
import re
from tqdm import tqdm
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings



from typing import List, Dict


os.environ['QDRANT_HOST'] = "https://21400720-991d-4ca8-8191-e220f5b545ce.us-east4-0.gcp.cloud.qdrant.io:6333"
os.environ['QDRANT_API_KEY'] = "BA8RYa_t2LUEWtYmnor7u9EC6GPIalvobOthYX3UJYZMPOhZkfpT4A"



client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True
)

collection_name = "test_collection_1"

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


db = Qdrant(client=client, embeddings=embeddings, collection_name=collection_name)

# for testing the right collection was queried
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
collection_info = qdrant_client.get_collection(collection_name=collection_name)
print(list(collection_info))

groq_client = None
LLAMA3_70B = "llama3-70b-8192"
LLAMA3_8B = "llama3-8b-8192"
GEMMA_7B_IT = "gemma-7b-it"

DEFAULT_MODEL = LLAMA3_70B

def setup_groq_client(groq_api_key, model_name=DEFAULT_MODEL):
    global groq_client
    groq_client = ChatGroq(temperature=0,
                           model_name=model_name,
                           groq_api_key=groq_api_key)


def generate_llm_response(chat_history):
    global groq_client

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        groq_client, db.as_retriever(), contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context/document to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know."
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. Note that we can also use StuffDocumentsChain and other
    # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(groq_client,
                                                         qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    response = rag_chain.invoke({"input": chat_history[-1]["content"],
                                 "chat_history": chat_history})
    return f"response: \n{response}\n\n\n chat_history: \n{chat_history}"
    # return response['answer']


def groq_chat_completion(urls: List[str],
                         session_messages: List[tuple],
                         doc_type: str = "general",
                         file_filter=""):
    chat_history = session_messages


    response = generate_llm_response(chat_history)
    return response
