# 🏥 Medical RAG System

A Retrieval-Augmented Generation (RAG) application built with **Streamlit** to answer medical-related queries using local NDJSON data and OpenAI’s language model.

---

## 🚀 Project Details
This project enables medical professionals, students, and researchers to query structured medical data.  
The system:
- Reads medical records stored in **NDJSON** format.  
- Embeds and indexes them using **FAISS** for fast retrieval.  
- Uses **LangChain** with **OpenAI API** to generate context-aware answers.  
- Provides an easy-to-use interface via **Streamlit**.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** – UI framework  
- **LangChain** – Orchestration framework for LLMs  
- **FAISS** – Vector database for document retrieval  
- **OpenAI GPT models** – LLM backend  
- **NDJSON** – Dataset format  
