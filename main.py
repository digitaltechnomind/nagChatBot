import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate


import re
import os

from dotenv import load_dotenv

load_dotenv()


def normalize_text(text: str) -> str:
    """
    Clean text for embedding
    """
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= normalize_text(page.extract_text())
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def addTo_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context.

    If the answer is not in the context, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | model
    return chain

def user_input(user_question):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = get_qa_chain()

    response = chain.invoke({
        "context": context,
        "question": user_question
    })

    st.write("Reply:", response)
    
def main():
    st.set_page_config("Chat PDF - saurabh.banerjee@nagarro.com")
    st.header("Chat with PDF Files from assignment ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
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
