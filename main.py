import config
import vectorstore_manager
import chatbot_engine

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Retrieves a chat history for a given session ID from Redis.
    The history is created automatically if it doesn't exist.
    """
    return RedisChatMessageHistory(session_id=session_id, url=config.REDIS_URL)

def main():
    """
    The main function to initialize and run the chatbot.
    """
    #
   # gemini model
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0.3)

    retriever = vectorstore_manager.get_retriever()

    conversational_rag_chain = chatbot_engine.create_chatbot_chain(llm, retriever)

    chat_with_history = RunnableWithMessageHistory(
        conversational_rag_chain,
        get_session_history,  # returns a Redis history object 
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("\n--- University CampusBot is Online (with Redis Memory) ---")
    print("Type 'quit' or 'exit' to end the session.")

    session_id = "student_natnael_session"

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("CampusBot: Goodbye! Your chat history is saved.")
                break

            print("CampusBot: ", end="", flush=True)
            # Invoke the chain, which now automatically handles Redis history
            response = chat_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(response['answer'])

        except KeyboardInterrupt:
            print("\nCampusBot: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
