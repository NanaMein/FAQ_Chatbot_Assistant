import os
import json
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.text_splitter import SentenceSplitter



from llama_index.core.memory import ChatMemoryBuffer, BaseMemory
from llama_index.core.chat_engine.types import ChatMessage, ChatMode
from llama_index.core.chat_engine import CondensePlusContextChatEngine


load_dotenv()

# embed_model = HuggingFaceEmbedding()
# llm = Groq(model=os.getenv('TEST_GROQ_SMALL'), api_key=os.getenv('GROQ_API_KEY'))
#
# documents = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()
#
# splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
#
# nodes = splitter.get_nodes_from_documents(documents)
#
# index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
#
# memory = ChatMemoryBuffer.from_defaults(token_limit=3000, chat_history=[])
#
# chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, memory=memory, llm=llm)
# # chat_engine_v1 = index.as_chat_engine(memory=memory, llm=llm, chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT)


embed_model = HuggingFaceEmbedding()
llm = Groq(model=os.getenv("TEST_GROQ_SMALL" ) or os.getenv("GROQ_MODEL"),
           api_key=os.getenv('GROQ_API_KEY')
           )

documents = SimpleDirectoryReader(input_dir=os.getenv('DATA_DIR')).load_data()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

nodes = splitter.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)

retriever = index.as_retriever()

memory = ChatMemoryBuffer.from_defaults(token_limit=3000, chat_history=[]) #chat_history=[]

chat_engine = CondensePlusContextChatEngine.from_defaults(
    llm=llm,
    memory=memory,
    system_prompt="you are a youthful and cheerful assitant",
    retriever=retriever,
) #chat_history=[]

# chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, memory=memory, llm=llm)
# chat_engine_v1 = index.as_chat_engine(memory=memory, llm=llm, chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT)

if __name__=='__main__':
    while True:
        loop_message = input("what do you want to say? ")
        loop_message_output =chat_engine.chat(loop_message)
        print(loop_message_output)



