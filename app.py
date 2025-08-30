import os
import json
import pandas as pd
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# ----------------- Helper Functions -----------------

def load_ndjson(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def preprocess_data(data):
    processed = []
    for item in data:
        processed.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "option_a": item.get("opa"),
            "option_b": item.get("opb"),
            "option_c": item.get("opc"),
            "option_d": item.get("opd"),
            "correct_answer": item.get("cop"),
            "explanation": item.get("exp"),
            "subject": item.get("subject_name"),
            "topic": item.get("topic_name"),
            "choice_type": item.get("choice_type")
        })
    return processed


def map_correct_answer(row):
    cop = str(row["correct_answer"]).strip() if row["correct_answer"] else None
    if cop in ["1", "2", "3", "4"]:
        return {
            "1": row["option_a"],
            "2": row["option_b"],
            "3": row["option_c"],
            "4": row["option_d"]
        }[cop]
    elif cop and cop.upper() in ["A", "B", "C", "D"]:
        return {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"]
        }[cop.upper()]
    else:
        return cop


def create_documents(df):
    docs = []
    for _, row in df.iterrows():
        text = f"""
        Question: {row['question']}
        Options:
          A. {row['option_a']}
          B. {row['option_b']}
          C. {row['option_c']}
          D. {row['option_d']}
        Correct Answer: {row['correct_answer']}
        Explanation: {row['explanation']}
        Subject: {row['subject']}
        Topic: {row['topic']}
        """
        metadata = {
            "id": row['id'],
            "subject": row['subject'],
            "topic": row['topic'],
            "choice_type": row['choice_type']
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ----------------- Streamlit App -----------------

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📘 RAG-powered Question Answering")

# Load dataset
file_path = "data/dev.json"
raw_data = load_ndjson(file_path)
processed_data = preprocess_data(raw_data)
df = pd.DataFrame(processed_data)
df["correct_answer_text"] = df.apply(map_correct_answer, axis=1)

# Create documents
docs = create_documents(df)

# Vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("vectorstore/faiss_index"):
    vectorstore = FAISS.load_local("vectorstore/faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local("vectorstore/faiss_index")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM (you must set OPENAI_API_KEY in your terminal or Streamlit secrets)
api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ----------------- User Interface -----------------
st.subheader("Ask a Question from your Dataset")

user_query = st.text_input("Enter your question:")
if st.button("Get Answer") and user_query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_query)
        st.success(response)














