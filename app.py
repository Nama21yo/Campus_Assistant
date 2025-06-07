import streamlit as st
import config  # Your existing config file
import vectorstore_manager  # Your existing vectorstore manager
import chatbot_engine  # Your existing chatbot engine

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Page Configuration ---
st.set_page_config(
    page_title="CampusBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching for Performance ---
# This decorator ensures that the heavy objects are loaded only once
@st.cache_resource
def get_rag_chain():
    
    # 2. Get the retriever
    retriever = vectorstore_manager.get_retriever()
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0.3)
    # 3. Create the conversational RAG chain
    conversational_rag_chain = chatbot_engine.create_chatbot_chain(llm, retriever)
    print("Backend: RAG chain initialized successfully.")
    return conversational_rag_chain

# --- Session State Initialization ---
# This ensures that the variables persist across user interactions
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = [] # For displaying in the UI
if "rag_chain_with_history" not in st.session_state:
    st.session_state.rag_chain_with_history = None

# --- UI Rendering ---

# Sidebar for session management
with st.sidebar:
    st.title("🎓 CampusBot")
    st.markdown("Your friendly AI assistant for all campus-related queries.")
    st.markdown("---")
    
    # Input for user name/session ID
    session_name = st.text_input("Enter your name to start a session:", key="session_name_input")
    
    if st.button("Start Chat", key="start_chat_button"):
        if session_name:
            # Sanitize the name to create a valid session ID
            st.session_state.session_id = f"student_session_{''.join(filter(str.isalnum, session_name))}"
            st.success(f"Session started for {session_name}!")
            
            # Re-initialize the chain with the new session history
            base_rag_chain = get_rag_chain()
            st.session_state.rag_chain_with_history = RunnableWithMessageHistory(
                base_rag_chain,
                lambda session_id: RedisChatMessageHistory(
                    session_id=session_id, url=config.REDIS_URL
                ),
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            # Clear UI history for the new session
            st.session_state.chat_history_ui = []
        else:
            st.warning("Please enter a name to start the chat.")

# Main chat interface
if not st.session_state.session_id:
    st.info("Please enter your name in the sidebar and click 'Start Chat' to begin.")
else:
    st.header(f"Chatting as: {st.session_state.get('session_name_input', 'Student')}")
    
    # "New Conversation" button to clear UI history but keep the same session ID
    if st.button("Start New Conversation", key="new_conversation_button"):
        st.session_state.chat_history_ui = []
        st.success("New conversation started. Your previous chat with Redis is saved.")
        # Optional: You could also clear the Redis history if you prefer
        # redis_history = RedisChatMessageHistory(session_id=st.session_state.session_id, url=config.REDIS_URL)
        # redis_history.clear()

    # Display existing chat history from the session state
    for message in st.session_state.chat_history_ui:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for new user prompts
    if prompt := st.chat_input("Ask me anything about the campus..."):
        # Add user message to the UI history
        st.session_state.chat_history_ui.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response by streaming
        with st.chat_message("assistant"):
            
            # The generator function that streams the response
            def stream_response_generator():
                response_stream = st.session_state.rag_chain_with_history.stream(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                for chunk in response_stream:
                    # Yield the 'answer' part of the chunk
                    content = chunk.get("answer", "")
                    yield content

            # Use st.write_stream to render the streaming response
            full_response = st.write_stream(stream_response_generator)
        
        # Add the complete assistant response to the UI history
        st.session_state.chat_history_ui.append({"role": "assistant", "content": full_response})
