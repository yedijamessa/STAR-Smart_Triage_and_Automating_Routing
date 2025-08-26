# 🌟 STAR: Smart Triage & Automated Routing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16953764.svg)](https://doi.org/10.5281/zenodo.16953764)

**STAR** (Smart Triage & Automated Routing) is an **LLM-powered triage system** for classifying, prioritising, and routing IT support tickets.  
It leverages **Large Language Models (LLMs)**, **retrieval-augmented generation (RAG)**, and explainable AI to provide automated support with transparency.

---

## 🚀 Features
- 🔍 **Automatic classification** of IT support tickets  
- ⏱ **Urgency & priority inference** (High vs Normal)  
- 🗂 **Intent and category detection**  
- 📚 **Similarity search** using vector embeddings (FAISS / RAG)  
- 🤖 **Automated responses & resolutions** powered by LLMs  
- 🧩 **Multi-agent pipeline**:
  - Understanding Agent  
  - Prioritisation Agent  
  - Routing Agent  
  - Resolution Agent  
  - Similarity Checker  

---

## 📂 Project Structure
```
.
├── backend/        # Python backend for ticket processing and LLM integration
├── frontend/       # React + TypeScript frontend interface
└── README.md       # Documentation
```

⚠️ **Note**: Some files are removed from this public repo for **privacy & security** reasons:
1. `tickets_cleaned.xlsx` → raw ticket data used for RAG experiments  
2. `ticket_faiss_index/` → generated FAISS vector index (derived from dataset)  
3. `.env` → API keys (e.g., GPT model keys)  

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yedijamessa/STAR-Smart_Triage_and_Automating_Routing.git
cd STAR-Smart_Triage_and_Automating_Routing
```

### 2. Backend Setup (Python)
```bash
cd backend
pip install -r requirements.txt
```

Run backend:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup (React + TypeScript)
```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Usage
1. Input a new ticket via the frontend UI.  
2. The backend processes it through the LLM pipeline:  
   **Understanding → Prioritisation → Routing → Resolution → Similar Ticket Retrieval**  
3. View the results in the frontend, including:  
   - Urgency Code  
   - Inferred Intent & Category  
   - Similar Tickets  
   - Suggested Resolution  

---

## 📊 Evaluation Metrics
The system can be evaluated on:
- **Classification Accuracy** → Category, Intent  
- **Urgency Prediction** → Precision, Recall, F1 (High class focus)  
- **Retrieval Quality** → Recall@k, Mean Cosine Similarity@k  
- **Resolution Relevance** → Cosine similarity with gold solution (mean ± std)  
- **Latency & Cost per Ticket** → Time and API token usage  

---

## 🔒 Privacy & Security
This project involves **sensitive IT support data**.  
To comply with **GDPR** and internal policies:
- No raw data files are included.  
- All API keys and credentials are excluded.  
- Only anonymised or synthetic examples may be added later.  

---

## 📌 Acknowledgements
- **Datel Group** – Industry Project (University of Manchester, MSc Data Science)  
- **OpenAI GPT models** for LLM-powered triage  
- **FAISS** for vector similarity search  
- **React + Vite + TypeScript** for frontend  

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
