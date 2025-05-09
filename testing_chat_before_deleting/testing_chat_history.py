from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1200)

chat_history = [
    ChatMessage(role="user", content="Hello, how are you?"),
    ChatMessage(role="assistant", content="I'm doing well, thank you!"),
]

# put a list of messages
memory.put_messages(chat_history)



# put one message at a time
memory.put_message(chat_history[0])

# Get the last X messages that fit into a token limit
history = memory.get()

# Get all messages
all_history = memory.get_all()

# clear the memory
memory.reset()


from pymilvus import connections, Collection
from dotenv import load_dotenv
import os

load_dotenv()



# Connect to Zilliz
connections.connect(
    alias="default",
    uri=os.getenv('CLUSTER_URI'),
    token=os.getenv('CLUSTER_TOKEN')
)

# Load your collection
collection = Collection("kokomi_collections")
collection.load()

# Query the first 3 records
results = collection.query(
    expr="",  # Empty string = no filter (get all)
    output_fields=["text", "file_name", "embedding"],  # Fields to return
    limit=8
)

for result in results:
    print(f"File: {result['file_name']}")
    print(f"Text snippet: {result['text'][:100]}...")  # First 100 chars
    print(f"Embedding dim: {len(result['embedding'])}")  # Should be 384
    print("----------------------------------------------------")