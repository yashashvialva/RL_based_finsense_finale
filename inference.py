import os
import json
from openai import OpenAI
from finsense.env import FinSenseEnv
from finsense.models import ActionModel
 
def get_fallback_action(obs: dict) -> ActionModel:
    """
    FIX 2: Fallback is now 'avoid' instead of 'allow'.
    When the LLM fails, we save money rather than spend it.
    """
    exp = obs.get("current_expense")
    necessity = exp.get("necessity_tag", "discretionary") if exp else "discretionary"
 
    # Only allow essential expenses as fallback, avoid everything else
    if necessity == "essential" and exp:
        return ActionModel(decision="allow", approved_amount=float(exp["amount"]), reasoning="Fallback: essential expense allowed")
    return ActionModel(decision="avoid", approved_amount=0.0, reasoning="Fallback: API error, avoiding non-essential")
 
 
def build_prompt(obs: dict) -> str:
    """
    FIX 3: Much stronger prompt that tells the agent it MUST save money.
    Includes explicit savings target so agent understands the goal.
    """
    exp = obs.get("current_expense", {}) or {}
    balance = obs.get("balance", 0)
    goal_remaining = obs.get("goal_remaining", 0)
    days_left = obs.get("days_left", 1)
    stress = obs.get("stress_level", 0)
    risk = obs.get("risk_level", "low")
    required_per_day = goal_remaining / max(1, days_left)
 
    necessity = exp.get("necessity_tag", "discretionary")
    amount = exp.get("amount", 0)
    name = exp.get("name", "Unknown")
    category = exp.get("category", "unknown")
 
    # FIX 4: Strong necessity guidance — essential can't be fully reduced
    necessity_guidance = {
        "essential": "This is ESSENTIAL. You MUST allow it at full amount. Do NOT reduce or avoid.",
        "semi-essential": "This is semi-essential. You may reduce it but avoid only if balance is critically low.",
        "discretionary": "This is DISCRETIONARY. You SHOULD avoid or heavily reduce this to save money."
    }.get(necessity, "")
 
    return f"""You are a strict financial agent. Your ONLY goal is to save ₹{goal_remaining:.0f} in {days_left} days.
 
CURRENT EXPENSE: {name} | Category: {category} | Amount: ₹{amount:.0f} | Necessity: {necessity}
{necessity_guidance}
 
FINANCIAL STATE:
  Balance: ₹{balance:.0f}
  Goal Remaining: ₹{goal_remaining:.0f}
  Days Left: {days_left}
  Required Savings Per Day: ₹{required_per_day:.0f}
  Stress Level: {stress:.2f} (avoid essential expenses raises stress, hurting your score)
  Risk Level: {risk}
 
RULES:
1. If necessity=essential → ALWAYS allow at full amount
2. If necessity=semi-essential → reduce by 30-50% or allow if balance is healthy
3. If necessity=discretionary → ALWAYS avoid unless balance is very comfortable
4. Never spend more than ₹{required_per_day * 1.5:.0f} in one step (your daily safe limit)
5. Your score depends on how much of the ₹{goal_remaining:.0f} goal you achieve
 
Respond ONLY with valid JSON, no markdown:
{{"decision": "allow|reduce|avoid", "approved_amount": <float>, "reasoning": "<one sentence>"}}"""
 
 
def run_inference():
    print("[START]")
 
    api_key = os.getenv("HF_TOKEN", "dummy_token")
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
 
    client = OpenAI(api_key=api_key, base_url=base_url)
 
    env = FinSenseEnv()
    obs = env.reset(task_id="easy")
 
    step_num = 0
    all_rewards = []
    done = False
 
    while not done:
        print("[STEP]")
        step_num += 1
 
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a financial decision agent. Always respond with raw valid JSON only, no markdown."},
                    {"role": "user", "content": build_prompt(obs)}
                ],
                max_tokens=200,
                temperature=0.2  # Lower temp = more consistent decisions
            )
            raw = response.choices[0].message.content.strip()
 
            # Strip markdown if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.split("```")[0].strip()
 
            action_dict = json.loads(raw)
 
            # FIX 4: Enforce essential expense rule — never reduce/avoid essential below full amount
            exp = obs.get("current_expense") or {}
            if exp.get("necessity_tag") == "essential" and action_dict.get("decision") in ("reduce", "avoid"):
                action_dict["decision"] = "allow"
                action_dict["approved_amount"] = float(exp.get("amount", 0))
                action_dict["reasoning"] = "Overridden: essential expense must be allowed"
 
            action = ActionModel(**action_dict)
            print(f" -> Agent decided to {action.decision} Rs. {action.approved_amount} - Reason: {action.reasoning}")
 
        except Exception as e:
            print(f" -> Encountered AI error: {e}")
            action = get_fallback_action(obs)
 
        obs, reward, done, info = env.step(action)
        all_rewards.append(round(reward, 4))
 
    rewards_str = ",".join(str(r) for r in all_rewards)
    goal_remaining = obs.get("goal_remaining", 0)
    goal_total = obs.get("goal_total", 1)
    success = goal_remaining <= 0
 
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")
 
 
if __name__ == "__main__":
    run_inference()