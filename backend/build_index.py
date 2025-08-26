from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Example documents
docs = [
    {"page_content": "Issue: password reset\nSolution: Reset via admin portal\nUrgency: High\nUrgency Code: 4"},
    {"page_content": "Issue: invoice mismatch\nSolution: Correct account mapping\nUrgency: Normal\nUrgency Code: 3"},
]

vectorstore = FAISS.from_texts([d["page_content"] for d in docs], embedding_model)
vectorstore.save_local("ticket_faiss_index")
print("✅ FAISS index built and saved.")
