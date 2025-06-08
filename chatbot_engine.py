from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

def create_chatbot_chain(llm, retriever):
    """
    Creates the complete conversational RAG chain with an improved prompt.
    """
    # This prompt for reformulating the question remains the same, it's very effective.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an expert AI assistant for the Addis Ababa University Help Desk, acting as a knowledgeable and reliable source of information. Your name is 'CampusBot'.\n\n"
        "Your core mission is to provide comprehensive, accurate, and easy-to-understand answers by strictly analyzing the context provided below. Follow these rules diligently:\n\n"
        "1.  **Synthesize, Don't Just List:** Do not simply extract phrases. Read and understand all the provided context, then synthesize the information into a coherent and detailed response. If multiple pieces of context are relevant, combine them into one complete answer.\n\n"
        "2.  **Structure for Clarity:**\n"
        "    *   For lists of items (e.g., courses, departments, streams), use bullet points (`- `).\n"
        "    *   For step-by-step instructions (e.g., withdrawal process, application steps), use a numbered list.\n"
        "    *   Use bold text (`**text**`) to highlight key terms, names, or deadlines.\n\n"
        "3.  **Be Thorough:** Always provide the most complete answer the context allows. If a user asks about a topic, explain it fully rather than giving a one-sentence reply.\n\n"
        "4.  **Adhere Strictly to Context:** Your knowledge is limited to the text provided in the 'Context' section. If the answer is not available in the context, you must explicitly state that you do not have the information and recommend the user visit the official university website or contact the relevant department. NEVER invent information.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
