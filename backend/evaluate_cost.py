import os
import uuid
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


from langchain_openai import ChatOpenAI
from v1_b_main import TriageSystem, Ticket, compute_similarity, USAGE, reset_usage, estimate_cost_usd


# ---------- helper: normalise urgency to {Low, Normal, High, Critical}


def normalize_urgency(u):
    if pd.isna(u):
        return None
    if isinstance(u, (int, float, np.integer, np.floating)):
        # map your numeric codes to labels here if you have them
        code = int(u)
        num_map = {1: "Low", 2: "Normal", 3: "High", 4: "Critical"}
        return num_map.get(code, "Normal")
    s = str(u).strip().lower()
    str_map = {
    "low": "Low",
    "normal": "Normal",
    "medium": "Normal",
    "high": "High",
    "urgent": "High",
    "critical": "Critical",
    "crit": "Critical",
    }
    return str_map.get(s, "Normal")

# ---------- load data


df = pd.read_excel("tickets_cleaned.xlsx").dropna(subset=["PROBLEM", "SOLUTION"]) # adjust column names if needed
has_category = "CATEGORY" in df.columns

# smaller sample to control cost/time (edit as needed)
N = int(os.getenv("EVAL_N", "50"))
if len(df) > N:
    test_df = df.sample(n=N, random_state=42)
else:
    test_df = df.copy()

# ---------- init system


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
triage = TriageSystem(llm)


# ---------- storage for metrics + usage/cost aggregation


y_true_cat, y_pred_cat = [], []
y_true_urg, y_pred_urg = [], []
retrieval_sims, res_sims, latencies = [], [], []


# cost/usage
run_chat_calls = 0
run_embed_calls = 0
run_chat_in_tokens = 0.0
run_chat_out_tokens = 0.0
run_embed_tokens = 0.0
run_cost_total = 0.0


# ---------- eval loop


for _, row in test_df.iterrows():
    ticket = Ticket(
        user_id="eval_user",
        timestamp=datetime.now(),
        channel="eval",
        description=str(row["PROBLEM"]),
        account=str(row.get("ACCOUNT", "eval")),
    )


    reset_usage() # per-ticket counters to zero


    t0 = time.perf_counter()
    result = triage.handle_ticket(ticket, str(uuid.uuid4()))
    latencies.append(time.perf_counter() - t0)


    # gold labels
    if has_category:
        y_true_cat.append(str(row["CATEGORY"]))
        y_pred_cat.append(result.category or "")


    gold_urg_label = normalize_urgency(row.get("URGENCYCODE"))
    gold_priority = "High" if gold_urg_label in ["High", "Critical"] else "Normal"
    y_true_urg.append(gold_priority)
    y_pred_urg.append(result.priority_level or "Normal") # internal level for fairness


    # retrieval similarity (generated resolution vs top-k solutions)
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


    # resolution similarity vs gold solution
    if result.resolution:
        try:
            res_sims.append(compute_similarity(result.resolution, str(row["SOLUTION"])))
        except Exception:
            pass


    # accumulate usage + estimated cost
    est = estimate_cost_usd() # uses USAGE counters from this ticket
    run_chat_calls += USAGE.chat_calls
    run_embed_calls += USAGE.embed_calls
    run_chat_in_tokens += est["input_tokens"]
    run_chat_out_tokens += est["output_tokens"]
    run_embed_tokens += est["approx_embed_tokens"]
    run_cost_total += est["total_cost"]

# ---------- compute metrics


if has_category:
    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    cat_f1 = f1_score(y_true_cat, y_pred_cat, average="macro")
else:
    cat_acc = None
    cat_f1 = None


prec, rec, f1, _ = precision_recall_fscore_support(
    y_true_urg, y_pred_urg, pos_label="High", average="binary"
)


mean_retrieval_sim = float(np.mean(retrieval_sims)) if retrieval_sims else 0.0
mean_res_sim = float(np.mean(res_sims)) if res_sims else 0.0
std_res_sim = float(np.std(res_sims)) if res_sims else 0.0
pct_above_060 = float(np.mean([s > 0.60 for s in res_sims]) * 100) if res_sims else 0.0
pct_above_075 = float(np.mean([s > 0.75 for s in res_sims]) * 100) if res_sims else 0.0


lat_mean = float(np.mean(latencies)) if latencies else 0.0
lat_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
lat_p90 = float(np.percentile(latencies, 90)) if latencies else 0.0


# ---------- print summary (metrics + cost)


print("=== RESULTS ===")
if has_category:
    print(f"Category Accuracy: {cat_acc:.2f}, Macro-F1: {cat_f1:.2f}")
else:
    print("Category metrics: skipped (no CATEGORY column).")
    print(f"Priority Precision (High): {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
    print(f"Mean Retrieval Similarity: {mean_retrieval_sim:.2f}")
    print(f"Resolution Similarity: mean={mean_res_sim:.2f}, std={std_res_sim:.2f}")
    print(f"% above 0.60: {pct_above_060:.1f}%, % above 0.75: {pct_above_075:.1f}%")
    print(f"Latency (s): mean={lat_mean:.2f}, p50={lat_p50:.2f}, p90={lat_p90:.2f}")


print("=== COST / USAGE (approx) ===")
print(f"Total chat calls: {run_chat_calls}, total embed calls: {run_embed_calls}")
print(f"Total input tokens≈{int(run_chat_in_tokens)}, output tokens≈{int(run_chat_out_tokens)}, embed tokens≈{int(run_embed_tokens)}")
if len(test_df):
    print(f"Avg per ticket → chat calls: {run_chat_calls/len(test_df):.2f}, embed calls: {run_embed_calls/len(test_df):.2f}")
    print(f"Avg per ticket → cost: ${run_cost_total/len(test_df):.5f}")
print(f"Run total cost: ${run_cost_total:.4f}")