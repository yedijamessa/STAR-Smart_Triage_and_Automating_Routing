# Goal: Convert 5275-ticket file into a RAG system using FAISS and OpenAI embeddings

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load XLSX
xlsx_path = "tickets_cleaned.xlsx"  # Replace with your actual file path
try:
    df = pd.read_excel(xlsx_path)
except Exception as e:
    print("Error reading XLSX:", e)
    raise

# Step 2: Drop rows missing critical fields
df = df.dropna(subset=["PROBLEM", "SOLUTION"])

# Step 3: Convert rows into Document objects
retrieval_docs = []
for _, row in df.iterrows():
    text = f"""Subject: {row['SUBJECT']}
Urgency: {row['URGENCYCODE']}
Problem: {row['PROBLEM']}
Solution: {row['SOLUTION']}"""
    retrieval_docs.append(Document(page_content=text))

# Step 4: Create FAISS vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(retrieval_docs, embedding_model)
vectorstore.save_local("ticket_faiss_index")
