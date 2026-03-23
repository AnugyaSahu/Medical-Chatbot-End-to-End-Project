from flask import Flask, render_template, request, jsonify
import os
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.prompt import *

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()

index_name = "medibot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)   

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k": 3})

chatModel = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

#default route
@app.route("/")
def index():
    return render_template("index.html")