import httpx
import json

base_url = "http://127.0.0.1:8000"

try:
    print("Resetting...")
    res = httpx.post(f"{base_url}/reset", json={"task_id": "easy", "seed": 42})
    print(res.status_code)
    print(res.json())
    
    print("\nStepping...")
    res = httpx.post(f"{base_url}/step", json={"decision": "allow", "approved_amount": 100.0, "reasoning": "test"})
    print(res.status_code)
    print(res.json())

except Exception as e:
    print("Error:", e)
