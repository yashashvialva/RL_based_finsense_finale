import httpx
import json
import time
import os
from openai import OpenAI

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_URL = "http://127.0.0.1:7860"  # Your FastAPI Server
LLM_BASE_URL = "http://localhost:11434/v1/"  # Ollama Local
LLM_MODEL = "mistral:latest"
LLM_API_KEY = "ollama"

# Initialize LLM Client
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def build_prompt(obs: dict) -> str:
    """Creates a concise prompt for the financial agent."""
    exp = obs.get("current_expense", {}) or {}
    name = exp.get("name", "Unknown")
    amount = exp.get("amount", 0)
    necessity = exp.get("necessity_tag", "discretionary")
    
    return f"""You are a Financial Planning Agent.
Current Status:
- Balance: ₹{obs.get('balance', 0):.0f}
- Goal Left: ₹{obs.get('goal_remaining', 0):.0f}
- Days Left: {obs.get('days_left', 0)}
- Stress: {obs.get('stress_level', 0.0):.2f}

Incoming Expense:
- Item: {name}
- Amount: ₹{amount:.0f}
- Necessity: {necessity}

Rules:
1. Always 'allow' essentials.
2. 'reduce' semi-essentials if budget is tight.
3. 'avoid' discretionary items if far from goal.

Reply ONLY with valid JSON: {{"decision": "allow/reduce/avoid", "approved_amount": float, "reasoning": "string"}}"""

def get_llm_decision(obs: dict):
    """Hits local Ollama to get a financial decision."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                {"role": "user", "content": build_prompt(obs)}
            ],
            temperature=0.1,
            max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()
            
        return json.loads(raw)
    except Exception as e:
        print(f"   ⚠️ LLM Error: {e}. Falling back to default.")
        exp = obs.get("current_expense") or {}
        return {
            "decision": "avoid", 
            "approved_amount": 0.0, 
            "reasoning": "Fallback due to LLM error"
        }

def test_task_flow(task_id):
    print(f"\n{'='*70}")
    print(f"🚀 PRODUCTION TEST: {task_id.upper()} (LLM-Driven)")
    print(f"{'='*70}")

    try:
        # 1. Reset Environment
        print(f"[*] /reset -> {task_id}...")
        reset_res = httpx.post(f"{BASE_URL}/reset", json={"task_id": task_id, "seed": 42})
        obs = reset_res.json()
        print(f"✅ Environment Ready. Balance: ₹{obs.get('balance'):.0f}")

        step_count = 0
        total_reward = 0.0
        done = False

        while not done:
            step_count += 1
            
            # 2. Get LLM Decision
            action_dict = get_llm_decision(obs)
            decision = action_dict.get("decision", "avoid")
            amount = action_dict.get("approved_amount", 0.0)
            reason = action_dict.get("reasoning", "No reason provided")

            # 3. Call /step API
            step_res = httpx.post(
                f"{BASE_URL}/step", 
                json={"decision": decision, "approved_amount": float(amount), "reasoning": reason}
            )
            
            if step_res.status_code != 200:
                print(f"❌ Step {step_count} failed!")
                break
                
            data = step_res.json()
            obs = data.get("observation", {})
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            total_reward += reward

            # Log current status
            exp = obs.get("current_expense") or {"name": "END OF DAY"}
            print(f"Step {step_count:3d} | Action: {decision:6s} | Item: {exp['name'][:15]:15s} | Rew: {reward:.2f} | Bal: {obs.get('balance'):.0f}")

            if step_count > 500: # Safety break
                print("⚠️ Safety break triggered (too many steps)")
                break

        # 4. Final Grading
        print(f"\n{'-'*70}")
        print(f"🏁 EPISODE FINISHED: {task_id}")
        print(f"Steps: {step_count} | Total Reward: {total_reward:.2f}")
        
        grade_res = httpx.post(f"{BASE_URL}/grade", json={"task_id": task_id})
        if grade_res.status_code == 200:
            print(f"🌟 FINAL SCORE: {grade_res.json().get('score'):.4f}")
        print(f"{'-'*70}")

    except Exception as e:
        print(f"❌ Critical Error testing {task_id}: {e}")

if __name__ == "__main__":
    print(f"Initializing Production API Test at {BASE_URL}...")
    print(f"Using LLM: {LLM_MODEL} at {LLM_BASE_URL}")

    # Ensure server is reachable
    try:
        httpx.get(f"{BASE_URL}/health")
    except:
        print(f"FATAL: Could not reach FastAPI server at {BASE_URL}. Is it running?")
        exit(1)

    # Test all tasks
    for task in ["easy", "medium", "hard"]:
        test_task_flow(task)
    
    print(f"\n{'='*70}")
    print("✨ PRODUCTION API TEST COMPLETE")
    print(f"{'='*70}")
