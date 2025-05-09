from langchain_groq import ChatGroq
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def worker_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv('TEST_GROQ_SMALL', 'GROQ_MODEL'),
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=.2,
    )

def manager_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv('TEST_GROQ_BIG', 'GROQ_MODEL'),
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=.5,
    )

def llama_llm() -> Groq:
    return Groq(
        model=os.getenv('TEST_GROQ_SMALL', 'GROQ_MODEL'),
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=.3,
    )