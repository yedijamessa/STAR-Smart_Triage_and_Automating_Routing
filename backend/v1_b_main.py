import json
import os
import datetime
import time

from dataclasses import dataclass
from typing import Dict, Optional, List
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np
load_dotenv()

USE_FAISS = os.getenv("USE_FAISS", "1") == "1"


# ===============================
# Cost & usage instrumentation
# (Steps 2–6 from your request)
# ===============================

@dataclass
class Usage:
    chat_calls: int = 0            # how many LLM chat calls were made
    embed_calls: int = 0           # how many embedding calls were made
    input_chars: int = 0           # total characters sent to LLM
    output_chars: int = 0          # total characters received from LLM

USAGE = Usage()

# --- Pricing (edit if your rates differ) ---
# Approx token conversion: tokens ≈ chars / 4
TOKENS_PER_CHAR = 1 / 4
# Example rates (USD per 1M tokens). Update to match your billing.
GPT4O_RATE_IN = 2.50 / 1_000_000     # $2.50 / 1M input tokens
GPT4O_RATE_OUT = 10.00 / 1_000_000   # $10.00 / 1M output tokens
EMBED_RATE = 0.13 / 1_000_000        # $0.13 / 1M tokens (text-embedding-3-large)


def reset_usage():
    USAGE.chat_calls = 0
    USAGE.embed_calls = 0
    USAGE.input_chars = 0
    USAGE.output_chars = 0


def estimate_cost_usd() -> Dict[str, float]:
    """Rough cost estimate for one ticket, based on counters.
    We approximate tokens from character length.
    """
    input_tokens = USAGE.input_chars * TOKENS_PER_CHAR
    output_tokens = USAGE.output_chars * TOKENS_PER_CHAR

    chat_in_cost = input_tokens * GPT4O_RATE_IN
    chat_out_cost = output_tokens * GPT4O_RATE_OUT

    # Embedding tokens are roughly proportional to characters embedded; we don't
    # track char lengths per embed in this simple setup, so use a conservative
    # average prompt length proxy based on output chars if available.
    # You can refine by passing actual text length into compute_similarity.
    approx_embed_tokens = max(1, int((USAGE.output_chars or 500) * TOKENS_PER_CHAR)) * USAGE.embed_calls
    embed_cost = approx_embed_tokens * EMBED_RATE

    total = chat_in_cost + chat_out_cost + embed_cost
    return {
        "chat_in_cost": chat_in_cost,
        "chat_out_cost": chat_out_cost,
        "embed_cost": embed_cost,
        "total_cost": total,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "approx_embed_tokens": approx_embed_tokens,
    }


# Load FAISS index for retrieval
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# simple in-memory embedding cache so we don't recompute for the same strings
_embed_cache: Dict[str, List[float]] = {}

def _embed_query_cached(text: str):
    if text in _embed_cache:
        return _embed_cache[text]
    emb = embedding_model.embed_query(text)
    USAGE.embed_calls += 1
    _embed_cache[text] = emb
    return emb


def extract_field(text, field_name):
    prefix = f"{field_name}:"
    for line in text.splitlines():
        if line.strip().lower().startswith(prefix.lower()):
            return line.split(":", 1)[1].strip()
    return ""


def compute_similarity(text1, text2):
    """Cosine similarity with cached embeddings + embed call counting."""
    emb1 = _embed_query_cached(text1)
    emb2 = _embed_query_cached(text2)
    # cosine similarity
    num = float(np.dot(emb1, emb2))
    den = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-12
    return num / den


FAISS_DIR = os.getenv("FAISS_DIR", "ticket_faiss_index")
vectorstore = None
if USE_FAISS:
    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


def find_similar_tickets(query: str, llm: ChatOpenAI, top_k: int = 3) -> List[Dict[str, str]]:
    """Return similar tickets via FAISS. If FAISS is disabled or unavailable,
    return an empty list so downstream logic continues safely."""
    print("[SimilarityChecker] Finding similar tickets...")

    # --- FAISS disabled or not loaded ---
    if not USE_FAISS or vectorstore is None:
        print("[SimilarityChecker] FAISS disabled or not loaded; skipping retrieval.")
        return []

    # --- Try retrieval; fail soft if index missing/corrupt ---
    try:
        retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"[SimilarityChecker] Retrieval error: {e}. Returning no similar tickets.")
        return []

    similar_results: List[Dict[str, str]] = []
    for doc in retrieved_docs:
        content = doc.page_content
        prompt = f"""
You're a helpful support assistant. Given a new ticket description:

NEW TICKET:
{query}

PAST TICKET:
{content}

Does the past ticket describe a similar issue? Reply only "Yes" or "No".
"""
        # --- cost counters ---
        USAGE.chat_calls += 1
        USAGE.input_chars += len(prompt)
        result = llm.invoke([HumanMessage(content=prompt)])
        USAGE.output_chars += len(result.content or "")
        # ----------------------
        response = (result.content or "").strip().lower()
        if "yes" in response:
            similar_results.append({
                "full": content,
                "solution": extract_field(content, "Solution"),
                "urgency": extract_field(content, "Urgency"),
                "urgency_code": extract_field(content, "Urgency Code"),
            })
    return similar_results



def infer_urgency_from_similar(llm: ChatOpenAI, similar: List[Dict[str, str]], ticket_text: str) -> str:
    # collect and normalize
    urg_list = []
    for s in similar:
        u = (s.get("urgency") or "").strip()
        if u:
            urg_list.append(u)

    uniq = {u.lower() for u in urg_list if u}

    # YOUR RULE: if all the urgency are the same -> let the LLM decide
    if len(uniq) == 1 and len(urg_list) > 0:
        prompt = f"""
Decide the urgency label for this ticket based on the description. Use only one of:
"Low", "Normal", "High", "Critical".

TICKET:
{ticket_text}

Answer with just the label.
"""
        # --- cost counters ---
        USAGE.chat_calls += 1
        USAGE.input_chars += len(prompt)
        res = llm.invoke([HumanMessage(content=prompt)])
        USAGE.output_chars += len(res.content or "")
        # ----------------------
        ans = (res.content or "").strip().lower()
        mapping = {"low":"Low","normal":"Normal","medium":"Normal","high":"High","critical":"Critical"}
        return mapping.get(ans, "Normal")

    # Otherwise (mixed or empty), infer from past tickets:
    # Majority vote over known labels (normalize to the same set)
    norm_map = {"low":"Low","normal":"Normal","medium":"Normal","high":"High","critical":"Critical"}
    counts = {}
    for raw in urg_list:
        key = norm_map.get(raw.lower())
        if key:
            counts[key] = counts.get(key, 0) + 1

    if counts:
        # pick max count; if tie, fall back to LLM
        sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_labels = [k for k,v in counts.items() if v == sorted_counts[0][1]]
        if len(top_labels) == 1:
            return top_labels[0]

    # tie or no urgency in similar → LLM decides
    prompt = f"""
Decide the urgency label for this ticket based on the description. Use only one of:
"Low", "Normal", "High", "Critical".

TICKET:
{ticket_text}

Answer with just the label.
"""
    # --- cost counters ---
    USAGE.chat_calls += 1
    USAGE.input_chars += len(prompt)
    res = llm.invoke([HumanMessage(content=prompt)])
    USAGE.output_chars += len(res.content or "")
    # ----------------------
    ans = (res.content or "").strip().lower()
    return {"low":"Low","normal":"Normal","medium":"Normal","high":"High","critical":"Critical"}.get(ans, "Normal")


def urgency_to_priority(urgency: str) -> str:
    u = (urgency or "").strip().lower()
    if u in ("critical", "high"):
        return "High"
    return "Normal"


def urgency_code_to_priority(code) -> str:
    try:
        c = int(str(code).strip())
    except Exception:
        return "Normal"
    if c in (5, 4):
        return "High"
    return "Normal"


def urgency_code_to_label(code) -> str:
    try:
        c = int(str(code).strip())
    except Exception:
        return "Standard Issue"
    return {
        5: "System Down",
        4: "Business Critical",
        3: "Standard Issue",
        6: "Standard Issue Customer",
        1: "Informational",
    }.get(c, "Standard Issue")


@dataclass
class Ticket:
    user_id: str
    timestamp: datetime
    channel: str
    description: str  # mapped from 'issue' in frontend
    account: str
    contact: Optional[str] = None
    project: Optional[str] = None
    area: Optional[str] = None
    input_category: Optional[str] = None


@dataclass
class ProcessedTicket:
    ticket_id: str
    raw_ticket: Ticket
    metadata: Dict
    intent: Optional[str] = None
    entities: Optional[dict] = None
    sentiment: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    priority_level: Optional[str] = None 
    assigned_to: Optional[str] = None
    auto_response: Optional[str] = None
    resolution: Optional[str] = None
    similar_tickets: Optional[List[Dict[str, str]]] = None
    resolution_similarity_score: float = 0.0
    inferred_urgency: Optional[str] = None
    inferred_urgency_code: Optional[int] = None


class IntakeAgent:
    def process(self, ticket: Ticket) -> dict:
        return {
            "account": ticket.account,
            "contact": ticket.contact,
            "project": ticket.project,
            "area": ticket.area,
            "input_category": ticket.input_category,
            "description": ticket.description
        }


class UnderstandingAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def analyze(self, ticket: Ticket) -> Dict:
        print("[UnderstandingAgent] Using LLM to analyze intent, entities, sentiment, and category...")
        prompt = f"""
        Analyze the following support ticket and return ONLY a valid JSON object with these keys:
        - intent (e.g., \"password reset\", \"billing issue\")
        - entities (a dictionary with a \"keywords\" list)
        - sentiment (e.g., \"positive\", \"neutral\", \"frustrated\")
        - category (e.g., \"Technical\", \"Billing\")

        Support ticket:
        {ticket.description}
        """
        # --- cost counters ---
        USAGE.chat_calls += 1
        USAGE.input_chars += len(prompt)
        result = self.llm.invoke([HumanMessage(content=prompt.strip())])
        USAGE.output_chars += len(result.content or "")
        # ----------------------
        try:
            content = (result.content or "").strip()
            if content.startswith("```json"):
                content = content[7:].strip()
            if content.startswith("```"):
                content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print("[Error parsing LLM response]", e)
            print("Raw content was:\n", result.content)
            return {}


class PrioritizationAgent:
    def prioritize(self, entities: Dict, sentiment: str) -> str:
        print("[PrioritizationAgent] Determining priority level...")
        keywords = entities.get("keywords", [])
        if "critical" in keywords or (sentiment or "").lower() == "frustrated":
            return "High"
        return "Normal"


class RoutingAgent:
    def route(self, category: str, priority: str) -> str:
        print("[RoutingAgent] Routing based on category and priority...")
        if category == "Billing":
            return "Billing Team"
        if priority == "High":
            return "Tier 2 Support"
        return "Tier 1 Support"


class KnowledgeAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def retrieve_response(self, query: str) -> str:
        print("[KnowledgeAgent] Generating auto-response using LLM...")
        prompt = f"Provide a helpful and concise support response to the following issue:\n\n{query}"
        # --- cost counters ---
        USAGE.chat_calls += 1
        USAGE.input_chars += len(prompt)
        result = self.llm.invoke([HumanMessage(content=prompt)])
        USAGE.output_chars += len(result.content or "")
        # ----------------------
        return (result.content or "").strip()


class ResolutionAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def resolve(self, query: str) -> str:
        print("[ResolutionAgent] Generating resolution using LLM only...")
        prompt = f"""
        Suggest a technical support resolution step for the following issue:

        {query}
        """
        # --- cost counters ---
        USAGE.chat_calls += 1
        USAGE.input_chars += len(prompt)
        result = self.llm.invoke([HumanMessage(content=prompt)])
        USAGE.output_chars += len(result.content or "")
        # ----------------------
        return (result.content or "").strip()


class FeedbackAgent:
    def monitor(self, processed_ticket: ProcessedTicket) -> None:
        print("[FeedbackAgent] Monitoring final outcome...")


class TriageSystem:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.intake = IntakeAgent()
        self.understanding = UnderstandingAgent(llm)
        self.prioritization = PrioritizationAgent()
        self.routing = RoutingAgent()
        self.knowledge = KnowledgeAgent(llm)
        self.resolution = ResolutionAgent(llm)
        self.feedback = FeedbackAgent()

    def handle_ticket(self, ticket: Ticket, ticket_id: str) -> ProcessedTicket:
        # Reset per-ticket usage counters
        reset_usage()

        timings: dict[str, float] = {}
        t_all = time.perf_counter()

        # 1) Intake
        t0 = time.perf_counter()
        metadata = self.intake.process(ticket)
        timings["intake_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2) Understanding (LLM)
        t0 = time.perf_counter()
        understanding = self.understanding.analyze(ticket)
        timings["understanding_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3) Priority (rule-based first)
        t0 = time.perf_counter()
        priority = self.prioritization.prioritize(
            understanding.get("entities", {}),
            understanding.get("sentiment", "") or ""
        )
        timings["prioritisation_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4) Routing
        t0 = time.perf_counter()
        assigned_to = self.routing.route(understanding.get("category", "") or "", priority)
        timings["routing_ms"] = (time.perf_counter() - t0) * 1000.0

        # 5) Knowledge (customer-facing)
        t0 = time.perf_counter()
        auto_response = self.knowledge.retrieve_response(ticket.description)
        timings["knowledge_ms"] = (time.perf_counter() - t0) * 1000.0

        # 6) Resolution (engineer-facing)
        t0 = time.perf_counter()
        resolution = self.resolution.resolve(ticket.description)
        timings["resolution_ms"] = (time.perf_counter() - t0) * 1000.0

        # 7) Similar tickets + similarity score + inferred urgency
        t0 = time.perf_counter()
        similar_texts = find_similar_tickets(ticket.description, self.llm, top_k=3)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

        # 7a) Compute similarity (embed+cosine)
        t0 = time.perf_counter()
        sims = []
        for s in similar_texts:
            sol = s.get("solution", "")
            if sol:
                sims.append(compute_similarity(resolution, sol))
        avg_sim = (sum(sims) / len(sims)) if sims else 0.0
        timings["similarity_ms"] = (time.perf_counter() - t0) * 1000.0

        # 7b) Infer urgency (consensus/LLM)
        t0 = time.perf_counter()
        inferred_urgency = infer_urgency_from_similar(self.llm, similar_texts, ticket.description)
        timings["urgency_infer_ms"] = (time.perf_counter() - t0) * 1000.0

        # Default internal level in case no code overrides happen later
        priority_level = urgency_to_priority(inferred_urgency or "")

        # If textual urgency exists, adjust routing to the internal level
        if inferred_urgency:
            assigned_to = self.routing.route(understanding.get("category", "") or "", priority_level)

        # 7c) Urgency-code based priority override (code wins when present)
        t0 = time.perf_counter()
        codes = []
        for s in similar_texts:
            code_raw = (s.get("urgency_code") or "").strip()
            if code_raw:
                try:
                    codes.append(int(code_raw))
                except Exception:
                    pass

        inferred_urgency_code = None
        if codes:
            # 5/4 most severe
            severity = {5: 2, 4: 2, 3: 1, 6: 1, 1: 0}
            inferred_urgency_code = max(codes, key=lambda c: severity.get(c, 0))

            # Compute both: internal level for routing + label for display
            priority_level = urgency_code_to_priority(inferred_urgency_code)  # "High"/"Normal"
            priority_label = urgency_code_to_label(inferred_urgency_code)     # "System Down"/...

            # use level for routing
            assigned_to = self.routing.route(understanding.get("category", "") or "", priority_level)
            # and set the outward-facing priority to the label
            priority = priority_label
        else:
            # no code found; keep whatever you had already
            pass
        timings["urgency_code_override_ms"] = (time.perf_counter() - t0) * 1000.0

        # Include timings & cost in metadata (non-breaking change)
        timings["total_ms"] = (time.perf_counter() - t_all) * 1000.0
        cost = estimate_cost_usd()
        try:
            meta = metadata if isinstance(metadata, dict) else {"raw_metadata": metadata}
            meta["timings"] = timings
            meta["usage"] = {
                "chat_calls": USAGE.chat_calls,
                "embed_calls": USAGE.embed_calls,
                "input_chars": USAGE.input_chars,
                "output_chars": USAGE.output_chars,
            }
            meta["cost_estimate_usd"] = cost
        except Exception:
            meta = {"timings": timings, "usage": USAGE.__dict__, "cost_estimate_usd": cost}

        processed = ProcessedTicket(
            ticket_id=ticket_id,
            raw_ticket=ticket,
            metadata=meta,
            intent=understanding.get("intent"),
            entities=understanding.get("entities"),
            sentiment=understanding.get("sentiment"),
            category=understanding.get("category"),
            assigned_to=assigned_to,
            auto_response=auto_response,
            resolution=resolution,
            similar_tickets=similar_texts,
            resolution_similarity_score=avg_sim,
            inferred_urgency=inferred_urgency,
            inferred_urgency_code=inferred_urgency_code,
            priority=priority,
            priority_level=priority_level,
        )

        print(
            "[Timings] intake={:.0f}ms, understand={:.0f}ms, prior={:.0f}ms, route={:.0f}ms, "
            "knowledge={:.0f}ms, resolution={:.0f}ms, retrieval={:.0f}ms, similarity={:.0f}ms, "
            "urg_infer={:.0f}ms, code_override={:.0f}ms, total={:.0f}ms".format(
                timings.get("intake_ms", 0),
                timings.get("understanding_ms", 0),
                timings.get("prioritisation_ms", 0),
                timings.get("routing_ms", 0),
                timings.get("knowledge_ms", 0),
                timings.get("resolution_ms", 0),
                timings.get("retrieval_ms", 0),
                timings.get("similarity_ms", 0),
                timings.get("urgency_infer_ms", 0),
                timings.get("urgency_code_override_ms", 0),
                timings.get("total_ms", 0),
            )
        )
        print(f"[Analysis] Avg. Resolution Similarity to Past: {avg_sim:.2f}")
        print(
            f"[Cost] chat_calls={USAGE.chat_calls} embed_calls={USAGE.embed_calls} "
            f"input_tokens≈{int(cost['input_tokens'])} output_tokens≈{int(cost['output_tokens'])} "
            f"embed_tokens≈{int(cost['approx_embed_tokens'])} total_cost≈${cost['total_cost']:.5f}"
        )

        self.feedback.monitor(processed)
        return processed


# (Optional) local manual test remains omitted/unchanged to avoid schema drift.
