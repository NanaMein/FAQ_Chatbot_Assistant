from llama_index_config import run_chatbot


def run():


    message = input("Hello what can i do for you?" )
    message_output = run_chatbot(message)
    print(message_output)


    while True:
        loop_message = input("Do you have any other questions? ")
        loop_message_output = run_chatbot(loop_message)
        print(loop_message_output)


if __name__=='__main__':
    run()

