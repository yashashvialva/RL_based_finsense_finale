"""
Local rule-based inference for testing FinSense without API credits.
Usage: python inference_local.py
"""
from finsense.env import FinSenseEnv
from finsense.models import ActionModel


def rule_based_agent(obs: dict) -> ActionModel:
    """Simple heuristic agent that makes reasonable financial decisions."""
    expense = obs.get("current_expense")
    if not expense:
        return ActionModel(decision="avoid", approved_amount=0.0, reasoning="No expense")

    amount = float(expense["amount"])
    necessity = expense["necessity_tag"]
    balance = obs.get("balance", 0)
    days_left = obs.get("days_left", 1)
    goal_remaining = obs.get("goal_remaining", 0)
    stress = obs.get("stress_level", 0)

    daily_budget = balance / max(1, days_left)

    # Essential: always allow (but reduce if budget is tight)
    if necessity == "essential":
        if amount <= daily_budget * 0.5:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Essential and affordable")
        else:
            reduced = min(amount, daily_budget * 0.4)
            return ActionModel(decision="reduce", approved_amount=reduced,
                               reasoning="Essential but over budget, reducing")

    # Semi-essential: allow if cheap, reduce if moderate, avoid if expensive
    if necessity == "semi-essential":
        if stress > 0.6:
            # High stress — allow to reduce stress buildup
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Allowing semi-essential to manage stress")
        if amount <= daily_budget * 0.2:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Semi-essential and cheap enough")
        elif amount <= daily_budget * 0.4:
            reduced = amount * 0.5
            return ActionModel(decision="reduce", approved_amount=reduced,
                               reasoning="Semi-essential, reducing spend")
        else:
            return ActionModel(decision="avoid", approved_amount=0.0,
                               reasoning="Semi-essential but too expensive")

    # Discretionary: mostly avoid, allow only if very cheap and goal is on track
    if necessity == "discretionary":
        savings_per_day_needed = goal_remaining / max(1, days_left)
        surplus = daily_budget - savings_per_day_needed
        if amount <= surplus * 0.1 and surplus > 0:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Discretionary but very cheap, surplus exists")
        else:
            return ActionModel(decision="avoid", approved_amount=0.0,
                               reasoning="Discretionary — skipping to save")

    return ActionModel(decision="avoid", approved_amount=0.0, reasoning="Fallback: avoid")


def run_local_inference():
    print("[START] Running local rule-based agent (no API needed)\n")

    env = FinSenseEnv()

    for task_id in ["easy", "medium", "hard"]:
        obs = env.reset(task_id=task_id)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = rule_based_agent(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"\n  [RESULT] task={task_id} | steps={steps} | "
              f"total_reward={total_reward:.2f} | "
              f"goal_left={obs['goal_remaining']:.0f}/{obs['goal_total']:.0f} | "
              f"final_balance={obs['balance']:.0f} | "
              f"stress={obs['stress_level']:.2f}")
        print(f"{'='*60}\n")

    print("[END]")


if __name__ == "__main__":
    run_local_inference()
