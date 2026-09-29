import streamlit as st
from langhchain_community.document_loaders import PyPDFLOader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

@st.cache_resource
def create_vectorstore():
    loader = PyPDFLOader("anand__resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunks_size = 500,
        chunk_overlap = 50

    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vectorstore
@st.cache_resource
def create_llm():
    pipe = pipeline(
        "text-generation",
        model="openai/gpt-oss-20b",
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True
    
    )
    llm = HuggingFacePipeline(
        pipeline = pipe
    )
    return llm

def answer_question(question, vectorstore, llm):
    documents = vectorstore.similarity_search(
        question,
        k=3
    )
    context ="\n\n".join(
        document.page_content
        for document in documents
    )
prompt = f """"
Your a resume assistant.

Answer the users questions ONLY using the information provided in the resume context below.

If the answer is not present in the context, say:
"I couldnt find the information in the resume."

Do not makeup information.

Resume Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)
return response

st.title("RAG RESUME CHATBOT"")
st.write("Ask questions from resume")

vectorstore = create_vectorstore()
llm = create_llm()
question = st.text_input("ASK A QUESTION")
if question:

    with st.spinner("Searching resume..."):

        answer = answer_question(
            question,
            vectorstore,
            llm
        )

    st.subheader("Answer")

    st.write(answer)