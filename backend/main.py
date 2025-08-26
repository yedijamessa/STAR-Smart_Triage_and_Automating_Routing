from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from v1_b_main import (
    TriageSystem,
    Ticket,
    find_similar_tickets,
    compute_similarity,
    infer_urgency_from_similar,
    urgency_to_priority,
    urgency_code_to_priority,
    urgency_code_to_label,  
)

from fastapi import FastAPI, Request, Query
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

load_dotenv()

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic request models ===
class TicketRequest(BaseModel):
    # Required
    account: str
    issue: str
    # Optional
    contact: str | None = None
    project: str | None = None
    area: str | None = None
    input_category: str | None = None
    # Defaults for compatibility
    user_id: str = "anon"
    channel: str = "web"

class TicketItem(BaseModel):
    # Required
    account: str
    issue: str
    # Optional
    contact: str | None = None
    project: str | None = None
    area: str | None = None
    input_category: str | None = None
    # Defaults for compatibility
    user_id: str = "anon"
    channel: str = "web"

class BatchRequest(BaseModel):
    tickets: list[TicketItem]

# === LLM + Triage system ===
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

triage = TriageSystem(llm)

# === REST: single (JSON) ===
@app.post("/api/triage")
async def triage_ticket(req: TicketRequest):
    ticket_id = f"TCK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    ticket = Ticket(
        user_id=req.user_id,
        timestamp=datetime.utcnow(),
        channel=req.channel,
        description=req.issue,           # map Issue -> description
        account=req.account,
        contact=req.contact,
        project=req.project,
        area=req.area,
        input_category=req.input_category
    )

    result = triage.handle_ticket(ticket, ticket_id)

    return {
        "ticket_id": ticket_id,
        "intent": result.intent,
        "entities": result.entities,
        "sentiment": result.sentiment,
        "category": result.category,
        "priority": result.priority,
        "assigned_to": result.assigned_to,
        "auto_response": result.auto_response,
        "resolution": result.resolution,
        "similar_tickets": result.similar_tickets,
        "resolution_similarity_score": result.resolution_similarity_score,
        "inferred_urgency": result.inferred_urgency
    }

# === REST: streaming (SSE) ===
@app.get("/api/triage-stream")
async def triage_stream(
    request: Request,
    # Defaults kept for compatibility
    user_id: str = Query("anon"),
    channel: str = Query("web"),
    # New required fields
    account: str = Query(..., description="Client/Department"),
    issue: str = Query(..., description="Issue description"),
    # Optional extras
    contact: str | None = Query(None),
    project: str | None = Query(None),
    area: str | None = Query(None),
    input_category: str | None = Query(None)
):
    # Create ID + Ticket
    ticket_id = f"TCK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    ticket = Ticket(
        user_id=user_id,
        timestamp=datetime.utcnow(),
        channel=channel,
        description=issue,            # map Issue -> description
        account=account,
        contact=contact,
        project=project,
        area=area,
        input_category=input_category
    )

    async def gen():
        # 1) Intake
        meta = triage.intake.process(ticket)
        yield {"step": "intake", "data": meta}

        # 2) Understanding (LLM)
        understanding = triage.understanding.analyze(ticket)
        yield {"step": "understanding", "data": understanding}

        # 3) Priority
        priority = triage.prioritization.prioritize(
            understanding.get("entities", {}),
            understanding.get("sentiment", "")
        )
        yield {"step": "prioritization", "data": {"priority": priority}}

        # 4) Routing
        assigned_to = triage.routing.route(understanding.get("category", ""), priority)
        yield {"step": "routing", "data": {"assigned_to": assigned_to}}

        # 5) Knowledge (LLM)
        auto_response = triage.knowledge.retrieve_response(ticket.description)
        yield {"step": "knowledge", "data": {"auto_response": auto_response}}

        # 6) Resolution (LLM)
        resolution = triage.resolution.resolve(ticket.description)
        yield {"step": "resolution", "data": {"resolution": resolution}}

        # 7) Similar tickets + similarity score
        similar = find_similar_tickets(ticket.description, triage.llm, top_k=3)

        # compute average resolution similarity (same logic as handle_ticket)
        sims = []
        for s in similar:
            sol = s.get("solution", "")
            if sol:
                sims.append(compute_similarity(resolution, sol))
        avg_sim = sum(sims)/len(sims) if sims else 0.0
        inferred_urgency = infer_urgency_from_similar(triage.llm, similar, ticket.description)

        if inferred_urgency:
            priority = urgency_to_priority(inferred_urgency)
            assigned_to = triage.routing.route(understanding.get("category",""), priority)

        # Urgency-code based priority override (code wins when present)
        codes = []
        for s in similar:
            code_raw = (s.get("urgency_code") or "").strip()
            if code_raw:
                try:
                    codes.append(int(code_raw))
                except Exception:
                    pass

        inferred_urgency_code = None
        priority_level = None
        if codes:
            severity = {5: 2, 4: 2, 3: 1, 6: 1, 1: 0}
            inferred_urgency_code = max(codes, key=lambda c: severity.get(c, 0))

            # internal level for routing; label for display
            priority_level = urgency_code_to_priority(inferred_urgency_code)   # "High"/"Normal"
            priority_label = urgency_code_to_label(inferred_urgency_code)      # "System Down"/...

            # re-route using level
            assigned_to = triage.routing.route(understanding.get("category",""), priority_level)

            # set outward-facing priority to label for the final payload
            priority = priority_label
        else:
            # Fall back if no code found
            priority_level = urgency_to_priority(inferred_urgency or "")

        yield {"step":"similar", "data": {
            "similar_tickets": similar,
            "resolution_similarity_score": avg_sim,
            "inferred_urgency": inferred_urgency,
            "inferred_urgency_code": inferred_urgency_code,
            "priority_level": priority_level
        }}


        # 8) Final aggregate (include ticket_id)
        yield {"step":"done", "data":{
            "ticket_id": ticket_id,
            "intent": understanding.get("intent"),
            "entities": understanding.get("entities"),
            "sentiment": understanding.get("sentiment"),
            "category": understanding.get("category"),
            "priority": priority,
            "priority_level": priority_level,
            "assigned_to": assigned_to,
            "auto_response": auto_response,
            "resolution": resolution,
            "similar_tickets": similar,
            "resolution_similarity_score": avg_sim,
            "inferred_urgency": inferred_urgency,
            "inferred_urgency_code": inferred_urgency_code
            
        }}
        await asyncio.sleep(0)

    async def event_source():
        async for m in gen():
            yield {
                "event": m["step"],
                "data": json.dumps(m["data"])
            }

    return EventSourceResponse(event_source())

# === REST: batch (JSON) ===
@app.post("/api/triage/batch")
async def triage_batch(req: BatchRequest):
    results = []
    for item in req.tickets:
        ticket_id = f"TCK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        ticket = Ticket(
            user_id=item.user_id,
            timestamp=datetime.utcnow(),
            channel=item.channel,
            description=item.issue,       # map Issue -> description
            account=item.account,
            contact=item.contact,
            project=item.project,
            area=item.area,
            input_category=item.input_category
        )
        r = triage.handle_ticket(ticket, ticket_id)
        results.append({
            "ticket_id": ticket_id,
            "user_id": item.user_id,
            "intent": r.intent,
            "entities": r.entities,
            "sentiment": r.sentiment,
            "category": r.category,
            "priority": r.priority,
            "assigned_to": r.assigned_to,
            "auto_response": r.auto_response,
            "resolution": r.resolution,
            "similar_tickets": r.similar_tickets,
            "resolution_similarity_score": r.resolution_similarity_score,
            "inferred_urgency": r.inferred_urgency,
            "inferred_urgency_code": r.inferred_urgency_code
        })
    return {"results": results}
