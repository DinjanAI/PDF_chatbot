import streamlit as st
import pdfplumber
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import os
import pandas as pd

import time
from openai import APIConnectionError


def embed_with_retry(**kwargs):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            return embeddings.client.create(**kwargs)
        except APIConnectionError as e:
            print(f"API Connection Error: {e}")
            retries += 1
            time.sleep(2)  # Add a delay before retrying
    # If all retries fail, handle the error or raise an exception
    print("Failed after multiple retries.")

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

counter = 0

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        # We need to split the text using Character Text Split such that it should not increase token size
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        document_search = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        with st.form("question_form"):
            query = st.text_input("Ask question..type exit if you have no questions..", key=f"query_{counter}")
            submit_button = st.form_submit_button("Submit")

        if query.lower() == 'exit':
            st.write('\nThanks for the conversation')
        elif submit_button:
            docs = document_search.similarity_search(query)
            answer = chain.run(input_documents=docs, question=query)
            st.write("Answer:\n", answer)


