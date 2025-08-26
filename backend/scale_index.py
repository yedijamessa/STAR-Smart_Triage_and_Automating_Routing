# scale_index.py  (run from backend/)
import pandas as pd, random
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

SRC_XLSX = "tickets_cleaned.xlsx"
OUT_DIRS = ["ticket_faiss_index_x2", "ticket_faiss_index_x5", "ticket_faiss_index_x10"]

df = pd.read_excel(SRC_XLSX).dropna(subset=["PROBLEM","SOLUTION"])
base = []
for _, r in df.iterrows():
    text = f"Subject: {r.get('SUBJECT','')}\nUrgency: {r.get('URGENCYCODE','')}\nProblem: {r['PROBLEM']}\nSolution: {r['SOLUTION']}"
    base.append(Document(page_content=text, metadata={"urgency_code": r.get("URGENCYCODE","")}))

def noisy(d: Document):
    tail = random.choice(["", " [context added]", " [variant]", " [dup]"])
    return Document(page_content=d.page_content + tail, metadata=d.metadata)

for mult, outdir in zip([2,5,10], OUT_DIRS):
    docs = base.copy()
    while len(docs) < len(base)*mult:
        docs.extend(noisy(d) for d in base)
    docs = docs[:len(base)*mult]
    print(f"Building {outdir} with {len(docs)} docs...")
    vs = FAISS.from_documents(docs, OpenAIEmbeddings())
    Path(outdir).mkdir(exist_ok=True)
    vs.save_local(outdir)
