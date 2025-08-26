import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from v1_b_main import TriageSystem, Ticket, compute_similarity
from langchain_openai import ChatOpenAI
from datetime import datetime

# --- 1) Helper: normalise urgency to {Low, Normal, High, Critical}
def normalize_urgency(u):
    if pd.isna(u):
        return None

    # numeric codes → map to labels (adjust if your codes differ)
    if isinstance(u, (int, float, np.integer, np.floating)):
        code = int(u)
        num_map = {
            1: "Low",
            2: "Normal",
            3: "High",
            4: "Critical"
        }
        return num_map.get(code, "Normal")

    # strings → map to labels
    s = str(u).strip().lower()
    str_map = {
        "low": "Low",
        "normal": "Normal",
        "medium": "Normal",
        "med": "Normal",
        "high": "High",
        "urgent": "High",
        "critical": "Critical",
        "crit": "Critical"
    }
    return str_map.get(s, "Normal")

# --- 2) Load data
df = pd.read_excel("tickets_cleaned.xlsx")
df = df.dropna(subset=["PROBLEM", "SOLUTION"])

# Optional: peek unique values to verify mappings
print("Unique URGENCYCODE values (first 20):", df["URGENCYCODE"].dropna().unique()[:20])

# If you have a CATEGORY column for “gold” category labels, keep it.
# Otherwise skip classification metrics that need it.
has_category = "CATEGORY" in df.columns

# Small test subset to control cost/time
test_df = df.sample(n=100, random_state=42) if len(df) > 100 else df.copy()

# --- 3) Initialise system
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
triage = TriageSystem(llm)

# --- 4) Storage
y_true_cat, y_pred_cat = [], []
y_true_urg, y_pred_urg = [], []
retrieval_sims = []
res_sims = []
latencies = []

# --- 5) Eval loop
for _, row in test_df.iterrows():
    ticket = Ticket(
        user_id="eval_user",
        timestamp=datetime.now(),
        channel="eval_channel",
        description=str(row["PROBLEM"])
    )

    t0 = time.time()
    result = triage.handle_ticket(ticket)
    latencies.append(time.time() - t0)

    # Classification (only if you have gold labels)
    if has_category:
        gold_cat = str(row["CATEGORY"])
        pred_cat = result.category or ""  # guard against None
        y_true_cat.append(gold_cat)
        y_pred_cat.append(pred_cat)

    # Priority mapping
    gold_urg_label = normalize_urgency(row.get("URGENCYCODE"))
    gold_priority = "High" if gold_urg_label in ["High", "Critical"] else "Normal"

    pred_priority = result.priority or "Normal"
    y_true_urg.append(gold_priority)
    y_pred_urg.append(pred_priority)

    # Retrieval similarity: compare generated resolution vs similar solutions (if present)
    if result.similar_tickets and result.resolution:
        sims = []
        for s in result.similar_tickets:
            sol = s.get("solution") or ""
            if sol:
                try:
                    sims.append(compute_similarity(result.resolution, sol))
                except Exception:
                    pass
        if sims:
            retrieval_sims.append(float(np.mean(sims)))

    # Resolution relevance: generated vs gold solution
    if result.resolution:
        try:
            res_sims.append(compute_similarity(result.resolution, str(row["SOLUTION"])))
        except Exception:
            pass

# --- 6) Metrics
# Classification (if applicable)
if has_category:
    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    cat_f1 = f1_score(y_true_cat, y_pred_cat, average="macro")
else:
    cat_acc = None
    cat_f1 = None

# Priority (High vs Normal)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true_urg, y_pred_urg, pos_label="High", average="binary"
)

# Retrieval similarity
mean_retrieval_sim = float(np.mean(retrieval_sims)) if retrieval_sims else 0.0

# Resolution similarity
mean_res_sim = float(np.mean(res_sims)) if res_sims else 0.0
std_res_sim  = float(np.std(res_sims)) if res_sims else 0.0
pct_above_060 = float(np.mean([s > 0.60 for s in res_sims]) * 100) if res_sims else 0.0
pct_above_075 = float(np.mean([s > 0.75 for s in res_sims]) * 100) if res_sims else 0.0

# Latency
lat_mean = float(np.mean(latencies)) if latencies else 0.0
lat_p50  = float(np.percentile(latencies, 50)) if latencies else 0.0
lat_p90  = float(np.percentile(latencies, 90)) if latencies else 0.0

# --- 7) Print
print("\n=== RESULTS ===")
if has_category:
    print(f"Category Accuracy: {cat_acc:.2f}, Macro-F1: {cat_f1:.2f}")
else:
    print("Category metrics: skipped (no CATEGORY column found).")
print(f"Priority Precision (High): {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
print(f"Mean Retrieval Similarity: {mean_retrieval_sim:.2f}")
print(f"Resolution Similarity: mean={mean_res_sim:.2f}, std={std_res_sim:.2f}")
print(f"% above 0.60: {pct_above_060:.1f}%, % above 0.75: {pct_above_075:.1f}%")
print(f"Latency (s): mean={lat_mean:.2f}, p50={lat_p50:.2f}, p90={lat_p90:.2f}")
