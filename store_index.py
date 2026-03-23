from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_docs = load_pdf_files("data/")
filtered_docs = filter_to_minimal_docs(extracted_docs)
texts_chunk = text_split(filtered_docs)
embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medibot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimensions,
        metric="cosine",
        serverless=ServerlessSpec(cloud="aws", region="us-west-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name,)