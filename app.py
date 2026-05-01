from flask import Flask, render_template, request, jsonify
import os
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_embeddings()
index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
chatModel = ChatOllama(model="llama3.2", base_url=OLLAMA_BASE_URL)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


def check_retrieved_chunks(user_input):
    """
    Directly retrieves chunks and checks if any were found.
    Returns (chunks, labeled_context) or (None, None) if empty.
    """
    retrieved_chunks = retriever.invoke(user_input)

    # Confidence check — if nothing retrieved, don't call LLM
    if len(retrieved_chunks) == 0:
        return None, None

    # Label chunks explicitly for traceability
    labeled_context = "\n\n".join([
        f"[Source chunk {i+1}]: {chunk.page_content}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    return retrieved_chunks, labeled_context


# Default route
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET", "POST"])
def get_bot_response():
    user_input = request.form["msg"]
    print("User input:", user_input)

    # Step 1 — Check retrieved chunks before calling LLM
    retrieved_chunks, labeled_context = check_retrieved_chunks(user_input)

    if retrieved_chunks is None:
        return "No relevant information found in my knowledge base. Please consult a qualified healthcare professional."

    # Step 2 — Log chunks for debugging (you can remove in production)
    print(f"Retrieved {len(retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:100]}...")

    # Step 3 — Run RAG chain normally
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]

    print("Bot response:", answer)
    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)