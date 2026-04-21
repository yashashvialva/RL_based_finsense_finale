"""Quick test: verify shock events + delayed consequences + graders all work."""
import logging
logging.basicConfig(level=logging.INFO)

from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel
from finsense.graders import grade_episode
from finsense.tasks import TASKS

env = FinSenseEnv()

for tid in ["easy", "medium", "hard"]:
    obs = env.reset(tid)
    done = False
    steps = 0
    while not done:
        exp = obs.get("current_expense") or {}
        n = exp.get("necessity_tag", "discretionary")
        a = float(exp.get("amount", 0))
        if n == "essential":
            act = ActionModel(decision="allow", approved_amount=a)
        elif n == "semi-essential":
            act = ActionModel(decision="reduce", approved_amount=round(a * 0.5, 2))
        else:
            act = ActionModel(decision="avoid", approved_amount=0.0)
        obs, r, done, info = env.step(act)
        steps += 1

    s = env.state
    t = TASKS[tid]
    sm = StateModel(
        current_day=t.days - s["days_left"],
        total_days=t.days,
        balance=s["balance"],
        initial_goal=s["goal_total"],
        current_goal_remaining=s["goal_remaining"],
        stress_level=s["stress_level"],
        risk_level=s["risk_level"],
        seed=42,
        task_id=tid,
        expected_fixed_expenses=s["expected_fixed_expenses"],
        income_shock_active=s["income_shock_active"],
        recent_spending=s["recent_spending"],
        user_type="balanced",
        current_expense_idx=0,
        daily_expenses=[],
        daily_expense_idx=0,
        terminated=True,
        truncated=False,
    )
    score = grade_episode(sm)
    score = max(0.01, min(0.99, score))
    gl = s["goal_remaining"]
    st = s["stress_level"]
    sh = s["income_shock_active"]
    pc = len(env.pending_consequences)
    print(f"\n>> {tid}: score={score:.4f} steps={steps} goal_left={gl:.0f} stress={st:.2f} shock={sh} pending={pc}")
    assert 0.01 <= score <= 0.99, f"Score out of range: {score}"

print("\n=== ALL TESTS PASSED ===")
