
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
