from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_postgres.vectorstores import PGVector
import requests
import os
import requests
import json
import httpx
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from datetime import date
from langchain_nomic import NomicEmbeddings
import warnings
import ssl
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
import re
from dotenv import load_dotenv

load_dotenv()
# -----------------------------
# Model setup used for embedding and chat
# -----------------------------
API_BASE = os.getenv("API_BASE") + "/v1"
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

def embed_text(text: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,   # make sure MODEL_NAME is a str, not a tuple
        "input": text
    }

    print("HEADERS:", headers)
    print("PAYLOAD:", payload)

    resp = requests.post(
        f"{API_BASE}/embeddings",
        headers=headers,
        json=payload,
        verify=False
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

class CustomEmbeddings:
    def embed_documents(self, texts): return [embed_text(t) for t in texts]
    def embed_query(self, text): return embed_text(text)

embedding = CustomEmbeddings()

# -----------------------------
# Vectorstore setup
# -----------------------------
# -----------------------------
# Database connection
# -----------------------------

DB_URI = os.getenv("DATABASE_URL")

vectorstore = PGVector(
    embeddings=embedding,
    connection=DB_URI,
    collection_name="maintenance_and_taxonomy",
    use_jsonb=True,
    create_extension=False,   # already created
)

httpx_client = httpx.Client(verify=False)

# Initialize LLM with credentials from cfenv
llm = ChatOpenAI(
    temperature=0.9,
    model=MODEL_NAME,
    base_url=API_BASE,
    api_key=API_KEY,
    http_client=httpx_client
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# -----------------------------
# Gradio chatbot function
# -----------------------------
def sanitize_answer(answer):
    # Convert sets to lists
    if isinstance(answer, set):
        answer = list(answer)
    # Convert anything else not string/dict/list to string
    if not isinstance(answer, (str, dict, list)):
        answer = str(answer)
    # Remove <think> tags if present
    if isinstance(answer, str):
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer

def predict(message, history):
    # Run RetrievalQA
    output = qa.invoke({"query": message})  # dict: {"result", "source_documents"}
    answer = sanitize_answer(output["result"])
    return answer
    
demo = gr.ChatInterface(
    fn=predict,
    title="🛠 Aircraft Maintenance Chatbot",
    description="Ask questions about maintenance records or aircraft taxonomy."
)


demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    quiet=False
)
