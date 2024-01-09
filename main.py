import streamlit as st
from functions import *
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
# from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import openai
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize memory and LLM
# llm = OpenAIChat()
llm = ChatOpenAI()

# Initialize the embedder
embedder = OpenAIEmbeddings()

memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="query")

# Initialize or retrieve the conversation history from the session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit App
st.title("ChatYourDocs x RAG")
st.write("Retrieval-Augmented Generation (RAG) improves LLM prediction quality by using an external datastore at inference \
        time to build a richer prompt that includes some combination of context and relevant knowledge.")
st.write("They main technical difference is the use of vector databases (vector stores) instead of relationl db's or document db's for example.")

# File Upload
uploaded_files = st.file_uploader("Choose files to upload (PDF, DOC, TXT)", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)

read_files = []
documents = ["hello"]

for uploaded_file in uploaded_files:
    file_type = uploaded_file.type.split('/')[-1]  # Extract the file type from MIME type
    print(file_type)

    # Read file content based on type
    if file_type == "pdf":
        document_text = read_pdf(uploaded_file)
    elif file_type == "docx":
        document_text = read_docx(uploaded_file)
    elif file_type == "plain":
        document_text = uploaded_file.read().decode()

    try:
        documents.append(document_text)
        read_files.append(uploaded_file.name)
        # st.write(f"{uploaded_file.name} reading completed! Now you may ask a question.")
    except Exception as e:
        st.write(f"An unexpected error occurred while processing {uploaded_file.name}: {str(e)}")

# Add documents to the vector store
if documents:
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
    document_chunks = text_splitter.create_documents(documents)
    vectorstore = FAISS.from_documents(document_chunks, embedder)
    vectorstore.add_documents(document_chunks)
    retriever = vectorstore.as_retriever()

# Display read files
st.write("Documents read: ", ", ".join(read_files))

# Define the ChatPromptTemplate
template = """You are an AI assistant that will help with questions about provided documents. Answer the question based on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the processing chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# User Input
user_input = st.text_input("Enter your question:")

# Ask Button
if st.button("Ask"):
    if user_input:

        # Invoke the chain
        response = chain.invoke(user_input)
        conversation_entry = {
            "user_input": user_input,
            "response": "",  
        }
         # Add the entry to the conversation history
        st.session_state.conversation_history.append(conversation_entry)
        # Update the conversation entry with the response
        conversation_entry["response"] = response

        st.write(response)

# Display Conversation History
st.subheader("Conversation History")
for entry in st.session_state.conversation_history:
    st.markdown(f"**You:**\n{entry['user_input']}\n\n**Bot:**\n{entry['response']}")

# Clear Conversation History Button
if st.button("Clear Conversation History"):
    st.session_state.conversation_history = []
    st.write("Conversation history has been cleared.")

