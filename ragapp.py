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
from langchain.document_loaders import CSVLoader
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
load_dotenv()

# get db connection url
DATABASE_URL = os.getenv("DATABASE_URL")

## load documents
loader_maintenance = CSVLoader(file_path="maintenance.csv")
docs_maintenance = loader_maintenance.load()

loader_taxonomy = CSVLoader(file_path="aircrafttaxonomy.csv")
docs_taxonomy = loader_taxonomy.load()

all_docs = docs_maintenance + docs_taxonomy


embeddings = OpenAIEmbeddings()

# See docker command above to launch a postgres instance with pgvector enabled.
connection = os.getenv("DATABASE_URL")  # Uses psycopg3!
collection_name = "my_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
vector_store.add_documents(all_docs)
print(f"✅ Inserted {len(all_docs)} documents into the vectorstore!")


# Initialize LLM with credentials from cfenv
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Create a retriever from your vectorstore
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Build a RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

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
