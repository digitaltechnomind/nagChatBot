import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

import re
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Initialize sliding window memory
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += normalize_text(page.extract_text())
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)


def addTo_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_pdfHFindex")


# -------------------------------
# Format last 4 turns
# -------------------------------
def format_chat_history(chat_history):
    formatted = ""
    for q, a in chat_history:
        formatted += f"User: {q}\nAssistant: {a}\n"
    return formatted.strip()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------------------
# Hugging Face Model Setup
# -------------------------------
@st.cache_resource
def load_hf_model():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.3
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context and recent conversation history.

    If the answer is not in the context, say:
    "answer is not available in the context"

    Conversation History (last 4 turns):
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    llm = load_hf_model()
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return prompt | llm



def user_input(user_question):
    embeddings = get_embeddings()

    new_db = FAISS.load_local(
        "faiss_pdfHFindex",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    # Build RAG context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Sliding window memory
    chat_history = st.session_state.chat_history[-4:]
    formatted_history = format_chat_history(chat_history)

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context,
        "question": user_question,
        "chat_history": formatted_history
    })

    # Update memory
    st.session_state.chat_history.append((user_question, response))

    if len(st.session_state.chat_history) > 4:
        st.session_state.chat_history = st.session_state.chat_history[-4:]

    st.write("Reply:", response)


def main():
    st.set_page_config("Chat PDF - HF Model + Memory")
    st.header("Chat with PDF Files (Hugging Face + Last 4 Turns Memory)")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                addTo_vector_store(text_chunks)
                st.success("Done")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()