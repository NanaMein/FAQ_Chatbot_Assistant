from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os

load_dotenv()

def llama_llm() -> Groq:
    return Groq(
        model=os.getenv('TEST_GROQ_SMALL'),
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=.3,
    )






def runner():
    while True:
        loop_message = input("what do you want to say? \n")
        chat_engine = SimpleChatEngine.from_defaults( llm=llama_llm())

        response = chat_engine.chat(loop_message)
        print("\n\n",response)

        if loop_message.lower() == "exit":
            print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
            print("Exiting the loop. Goodbye!")
            print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

            break  # Exit the loop



if __name__=='__main__':
    runner()