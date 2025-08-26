# sse_load_test.py
import asyncio, time, json, random
import aiohttp

API = "http://localhost:8000/api/triage-stream"
PAYLOADS = json.load(open("payloads.json"))

async def one(session, payload):
    params = { "account": payload["account"], "issue": payload["issue"] }
    t0 = time.perf_counter()
    try:
        async with session.get(API, params=params, timeout=120) as resp:
            if resp.status != 200:
                return time.perf_counter()-t0, False
            async for line in resp.content:
                if not line:
                    continue
                if b"event: done" in line:     # simplistic but works
                    return time.perf_counter()-t0, True
    except Exception:
        pass
    return time.perf_counter()-t0, False

async def run(concurrency=5, total=30):
    times, ok = [], 0
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [one(session, random.choice(PAYLOADS)) for _ in range(total)]
        for fut in asyncio.as_completed(tasks):
            dt, ok1 = await fut
            times.append(dt); ok += int(ok1)
    times.sort()
    def pct(p): return times[int(len(times)*p)]
    print(f"SSE Success={ok}/{total} Mean={sum(times)/len(times):.2f}s p50={pct(0.5):.2f}s p90={pct(0.9):.2f}s p99={pct(0.99):.2f}s")

if __name__ == "__main__":
    asyncio.run(run())
