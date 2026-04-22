"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
    MODEL_NAME = os.getenv("MODEL_NAME", "mistral:latest")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
import json
import sys
from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel, Expense
from finsense.graders import grade_episode
from finsense.tasks import TASKS
from finsense.memory import MemorySystem

# Import the local rule-based agent as fallback
from inference_local import rule_based_agent


def get_fallback_action(obs: dict, memory: MemorySystem = None, use_memory: bool = False) -> ActionModel:
    """
    Smart rule-based fallback with memory support.
    Used when API fails, LLM returns invalid output, or running without LLM.
    """
    return rule_based_agent(obs, memory=memory, use_memory=use_memory)


def extract_json(raw: str) -> str:
    """Sanitize raw LLM output to extract valid JSON."""
    raw = raw.strip()

    # Remove markdown code blocks
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

    # Extract first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    return raw


def build_prompt(obs: dict, memory_bias: str = None) -> str:
    """Strict JSON-enforcing prompt for financial decisions."""
    exp = obs.get("current_expense", {}) or {}
    necessity = exp.get("necessity_tag", "discretionary")
    amount = exp.get("amount", 0)
    name = exp.get("name", "Unknown")
    context = exp.get("context", "normal")
    balance = obs.get('balance', 0)
    goal_remaining = obs.get('goal_remaining', 0)
    days_left = obs.get('days_left', 0)
    active_events = obs.get("active_events", [])
    events_str = ", ".join(active_events) if active_events else "None"

    prompt = f"""You are a financial decision agent. Should I spend on this right now? Consider the balance, days left, event context, and memory.

Return ONLY valid JSON. No explanation. No markdown. No text before or after.

Format:
{{
  "decision": "allow" | "reduce" | "avoid",
  "approved_amount": number,
  "reasoning": "short string"
}}

Rules:
- essential → allow full
- semi-essential → reduce ~50%
- discretionary → avoid

Strict requirements:
- Use double quotes only
- No trailing commas
- No comments
- No extra text

Now decide:
Balance: {balance:.0f}
Goal Left: {goal_remaining:.0f}
Days Left: {days_left}
Active Events: {events_str}

Expense:
Name: {name}
Amount: {amount:.0f}
Type: {necessity}
Context: {context}"""

    if memory_bias:
        prompt += f"\nMemory suggests: {memory_bias} for similar situations (strong recommendation from past experience)."

    return prompt


def calculate_final_score(env, task_id):
    s = env.state
    task = TASKS.get(task_id)
    
    try:
        state_data = {
            "current_day": task.days - s["days_left"],
            "total_days": task.days,
            "balance": s["balance"],
            "initial_goal": s["goal_total"],
            "current_goal_remaining": s["goal_remaining"],
            "stress_level": s["stress_level"],
            "risk_level": s["risk_level"],
            "seed": 42,
            "task_id": task_id,
            "expected_fixed_expenses": s["expected_fixed_expenses"],
            "income_shock_active": s["income_shock_active"],
            "recent_spending": s["recent_spending"],
            "user_type": "balanced",
            "current_expense_idx": 0,
            "daily_expenses": [],
            "daily_expense_idx": 0,
            "terminated": s["days_left"] <= 0,
            "truncated": False
        }
        state_model = StateModel(**state_data)
        raw = grade_episode(state_model)
        # CRITICAL: always clamp, even if grader returns something unexpected
        return max(0.01, min(0.99, raw))
    except Exception:
        goal_total = s["goal_total"]
        goal_remaining = s["goal_remaining"]
        goal_progress = max(0.0, goal_total - goal_remaining)
        raw = goal_progress / max(1.0, goal_total)
        return max(0.01, min(0.99, raw))


def run_inference(task_id="easy"):
    # # Load environment variables
# No hardcoded keys in the default value!
 
    HF_TOKEN = os.getenv("HF_TOKEN", "ollama")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
    MODEL_NAME = os.getenv("MODEL_NAME", "mistral")

    use_memory = os.getenv("USE_MEMORY", "1") == "1"

    # Check if LLM is available
    llm_available = False
    client = None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        if MODEL_NAME:
            llm_available = True
    except Exception:
        pass

    env = FinSenseEnv()
    memory = env.memory

    obs = env.reset(task_id=task_id)

    model_label = MODEL_NAME if llm_available else "rule-based-fallback"
    print(f"[START] task={task_id} env=finsense-rl model={model_label}")

    if not llm_available:
        print(f"  [INFO] No LLM available - using rule-based agent with memory={'ON' if use_memory else 'OFF'}")

    step_num = 0
    all_rewards = []
    done = False
    last_error = "null"

    try:
        while not done:
            step_num += 1
            action_str = "null"
            
            try:
                # Get memory bias
                memory_bias = None
                if use_memory:
                    exp = obs.get("current_expense") or {}
                    expense_type = exp.get("category", "unknown")
                    context = exp.get("context", "normal")
                    necessity = exp.get("necessity_tag", "unknown")
                    active_events = obs.get("active_events", [])
                    event_type = active_events[0] if active_events else "none"
                    memory_bias = memory.get_memory_bias(
                        expense_type, context, event_type, necessity=necessity
                    )

                if llm_available and client and MODEL_NAME:
                    # Try LLM-based decision
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                            {"role": "user", "content": build_prompt(obs, memory_bias)}
                        ],
                        max_tokens=80,
                        temperature=0.1
                    )
                    raw = response.choices[0].message.content.strip()
                    print(f"[RAW LLM] {raw}")

                    # Safe JSON extraction
                    clean = extract_json(raw)
                    print(f"[CLEANED] {clean}")

                    # Fault-tolerant parsing
                    try:
                        action_dict = json.loads(clean)
                    except Exception:
                        # Hard fallback repair attempt
                        clean = clean.replace("'", '"')  # fix single quotes
                        clean = clean.replace("\n", " ")
                        try:
                            action_dict = json.loads(clean)
                        except Exception:
                            raise ValueError("JSON_PARSE_FAILED")

                    # Output validation guard
                    if action_dict.get("decision") not in ("allow", "reduce", "avoid"):
                        print(f"[VALIDATION] Invalid decision '{action_dict.get('decision')}' → forcing 'avoid'")
                        action_dict["decision"] = "avoid"
                    if not isinstance(action_dict.get("approved_amount"), (int, float)):
                        print(f"[VALIDATION] Invalid approved_amount → forcing 0.0")
                        action_dict["approved_amount"] = 0.0
                    
                    # Enforce essential rule
                    exp = obs.get("current_expense") or {}
                    if exp.get("necessity_tag") == "essential" and action_dict.get("decision") in ("reduce", "avoid"):
                        action_dict["decision"] = "allow"
                        action_dict["approved_amount"] = float(exp.get("amount", 0))

                    action = ActionModel(**action_dict)
                    action_str = f"{action.decision}({action.approved_amount:.0f})"
                    last_error = "null"
                else:
                    # Use rule-based agent with memory
                    action = get_fallback_action(obs, memory=memory, use_memory=use_memory)
                    action_str = f"{action.decision}({action.approved_amount:.0f})"
                    last_error = "null"

            except Exception as e:
                print(f"[PARSE ERROR] {e}")
                action = get_fallback_action(obs, memory=memory, use_memory=use_memory)
                action_str = f"fallback:{action.decision}({action.approved_amount:.0f})"
                last_error = str(e).replace('\n', ' ')[:200]

            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)

            # [STEP] line
            done_str = str(done).lower()
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={last_error}")

    except Exception as e:
        last_error = str(e).replace('\n', ' ')[:200]
    finally:
        success_bool = obs.get("goal_remaining", 0) <= 0
        success_str = str(success_bool).lower()
        score = calculate_final_score(env, task_id)
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

        # [END] line
        print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task)