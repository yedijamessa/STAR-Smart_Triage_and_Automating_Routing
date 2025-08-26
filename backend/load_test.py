# load_test.py
import asyncio, time, json, random, statistics
import httpx

API = "http://localhost:8000/api/triage"  # sync JSON endpoint
PAYLOADS = json.load(open("payloads.json"))

async def one(client, payload):
    t0 = time.perf_counter()
    try:
        r = await client.post(API, json=payload)
        dt = time.perf_counter() - t0
        ok = r.status_code == 200
        return dt, ok, (r.json() if ok else r.text)
    except Exception as e:
        return time.perf_counter()-t0, False, str(e)

async def run(concurrency=10, total=100):
    times, ok_count = [], 0
    async with httpx.AsyncClient(timeout=120) as client:
        sem = asyncio.Semaphore(concurrency)
        async def bound_one():
            async with sem:
                return await one(client, random.choice(PAYLOADS))
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(bound_one()) for _ in range(total)]
        for fut in asyncio.as_completed(tasks):
            dt, ok, _ = await fut
            times.append(dt)
            ok_count += int(ok)
        wall = time.perf_counter() - t0

    times.sort()
    def pct(p): return times[int(len(times)*p)]
    print(f"Concurrency={concurrency}  Total={total}")
    print(f"Success={ok_count}/{total} ({ok_count/total*100:.1f}%)")
    print(f"RPS={total/wall:.2f}/s  Mean={statistics.mean(times):.2f}s  p50={pct(0.5):.2f}s  p90={pct(0.9):.2f}s  p99={pct(0.99):.2f}s")

if __name__ == "__main__":
    import sys
    c = int(sys.argv[1]) if len(sys.argv)>1 else 10
    n = int(sys.argv[2]) if len(sys.argv)>2 else 100
    asyncio.run(run(concurrency=c, total=n))
