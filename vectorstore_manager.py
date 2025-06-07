import os
import config
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_retriever():
    """
    Creates and returns a Chroma vector store retriever.

    This function implements a "load-or-create" pattern. If a persistent
    vector store exists, it loads it. Otherwise, it creates one from
    the source documents and saves it to disk.
    """
    print("Initializing vector store retriever...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    if os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        print(f"Loading existing vector store from: {config.CHROMA_PERSIST_DIRECTORY}")
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        loader = TextLoader(config.DATA_FILE_PATH)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        print(f"Split data into {len(splits)} chunks.")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        print(f"New vector store created and saved to: {config.CHROMA_PERSIST_DIRECTORY}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Retriever with k=3 created successfully.")
    return retriever
