import os
import json
import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# ----------------- Helper Functions -----------------
def load_ndjson(uploaded_file):
    data = []
    for line in uploaded_file:
        line = line.decode("utf-8").strip()
        if line:
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
        return {"1": row["option_a"], "2": row["option_b"], "3": row["option_c"], "4": row["option_d"]}[cop]
    elif cop and cop.upper() in ["A", "B", "C", "D"]:
        return {"A": row["option_a"], "B": row["option_b"], "C": row["option_c"], "D": row["option_d"]}[cop.upper()]
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
            "id": row["id"],
            "subject": row["subject"],
            "topic": row["topic"],
            "choice_type": row["choice_type"]
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ----------------- Streamlit App -----------------
st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📘 RAG-powered Question Answering")

uploaded_file = st.file_uploader("Upload your NDJSON file", type=["json", "ndjson"])

if uploaded_file:
    raw_data = load_ndjson(uploaded_file)
    df = pd.DataFrame(preprocess_data(raw_data))
    df["correct_answer_text"] = df.apply(map_correct_answer, axis=1)

    docs = create_documents(df)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("vectorstore/faiss_index"):
        vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local("vectorstore/faiss_index")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    api_key = os.getenv("Open_ai_key")

    if not api_key:
        st.error("⚠️ OpenAI API key not found in Streamlit Secrets.")
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )

        prompt = PromptTemplate.from_template(
            """Use the context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
        )

        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.subheader("Ask a question")
        user_query = st.text_input("Enter your question:")

        if st.button("Get Answer") and user_query:
            with st.spinner("Thinking..."):
                response = chain.invoke(user_query)
                st.success(response)

else:
    st.info("⬆️ Please upload your NDJSON file to get started.")







