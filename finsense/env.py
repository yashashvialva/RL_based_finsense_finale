import logging
from typing import Dict, Any, Tuple
from finsense.models import ActionModel
from finsense.expense_generator import ExpenseGenerator
from finsense.tasks import TASKS

logger = logging.getLogger(__name__)


class FinSenseEnv:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.expense_gen = None
        self.daily_expenses = []
        self.expense_idx = 0
        self.daily_spend_total = 0.0
        self.daily_allowance = 0.0  # FIX 3: track allowance at START of day

    def reset(self, task_id: str = "easy", seed: int = 42) -> Dict[str, Any]:
        if task_id not in TASKS:
            task_id = "easy"

        task = TASKS[task_id]
        self.expense_gen = ExpenseGenerator(seed)

        # FIX 1: Balance = full salary. Fixed expenses are tracked separately, not pre-deducted.
        # This way balance doesn't go to zero immediately from large fixed costs.
        self.state = {
            "balance": task.initial_balance,
            "goal_total": task.goal,
            "goal_remaining": task.goal,
            "days_left": task.days,
            "stress_level": 0.0,
            "risk_level": "low",
            "expected_fixed_expenses": task.expected_fixed_expenses,
            "recent_spending": [],
            "avg_daily_spend": 0.0,
            "salary": task.initial_balance,
            "task_id": task_id,
            "income_shock_active": False,
        }

        self.daily_expenses = self.expense_gen.generate_daily_expenses(num_expenses=3)
        self.expense_idx = 0
        self.daily_spend_total = 0.0

        # FIX 3: Compute daily allowance = spendable money per day
        # (balance minus what needs to be saved, divided by days)
        self._recompute_daily_allowance()

        print(f"\n{'='*60}")
        print(f"  [RESET] task={task_id} | balance={self.state['balance']:.0f} | "
              f"goal={task.goal:.0f} | days={task.days}")
        print(f"  [INFO]  daily_allowance={self.daily_allowance:.0f} | "
              f"required_savings/day={task.goal / task.days:.0f}")
        print(f"{'='*60}")

        return self._get_observation()

    def _recompute_daily_allowance(self):
        """
        Daily allowance = how much the agent can spend TODAY and still hit the goal.

        Formula:
          spendable_total = balance - goal_remaining  (money not needed for goal)
          daily_allowance = spendable_total / days_left

        This is intentionally simple: if balance=27000, goal=5000, days=14
        → spendable = 22000 → allowance = 1571/day
        Agent can spend up to 1571 today and still save enough for the goal.
        """
        days_left = max(1, self.state["days_left"])
        balance = self.state["balance"]
        goal_remaining = self.state["goal_remaining"]

        spendable_total = max(0.0, balance - goal_remaining)
        self.daily_allowance = round(spendable_total / days_left, 2)

    def _get_observation(self) -> Dict[str, Any]:
        obs = dict(self.state)
        obs["recent_spending"] = obs["recent_spending"][-5:]
        days_left = max(1, self.state["days_left"])
        obs["required_savings_per_day"] = round(self.state["goal_remaining"] / days_left, 2)
        obs["goal_total"] = self.state["goal_total"]
        obs["daily_allowance"] = round(self.daily_allowance, 2)

        if self.expense_idx < len(self.daily_expenses):
            exp = self.daily_expenses[self.expense_idx]
            obs["current_expense"] = {
                "name": exp.name,
                "amount": exp.amount,
                "necessity_tag": exp.necessity_tag,
                "category": getattr(exp, "category", "unknown")
            }
        else:
            obs["current_expense"] = None

        return obs

    def step(self, action: ActionModel) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.state["days_left"] <= 0:
            return self._get_observation(), 0.0, True, {"error": "Environment is already done."}

        balance = self.state["balance"]
        spend = 0.0
        necessity = "none"

        if self.expense_idx < len(self.daily_expenses):
            current_expense = self.daily_expenses[self.expense_idx]
            necessity = current_expense.necessity_tag

            if action.decision == "allow":
                spend = current_expense.amount
            elif action.decision == "reduce":
                spend = min(action.approved_amount, current_expense.amount)
            elif action.decision == "avoid":
                spend = 0.0

            # FIX 1: Never spend more than available balance
            spend = min(spend, max(0.0, balance))

            self.state["balance"] -= spend
            self.daily_spend_total += spend

            # Stress update
            if action.decision == "avoid":
                if necessity == "essential":
                    self.state["stress_level"] += 0.3
                elif necessity == "semi-essential":
                    self.state["stress_level"] += 0.1
            elif action.decision == "allow" and necessity == "discretionary":
                # Mild stress relief for treating yourself, but not meaningful
                pass

        self.state["stress_level"] = min(1.0, self.state["stress_level"])

        # Track spending history
        recent = self.state["recent_spending"]
        recent.append(spend)
        self.state["recent_spending"] = recent[-5:]
        self.state["avg_daily_spend"] = sum(self.state["recent_spending"]) / max(1, len(self.state["recent_spending"]))

        # Risk level
        bal = self.state["balance"]
        fixed = self.state["expected_fixed_expenses"]
        if bal < fixed * 0.2:
            self.state["risk_level"] = "high"
        elif bal < fixed * 0.5:
            self.state["risk_level"] = "medium"
        else:
            self.state["risk_level"] = "low"

        # Per-step reward: penalise stress and risk
        stress_penalty = 0.4 * self.state["stress_level"]
        risk_penalty = {"low": 0.0, "medium": 0.1, "high": 0.3}[self.state["risk_level"]]
        reward = -stress_penalty - risk_penalty

        # Print expense line
        if self.expense_idx < len(self.daily_expenses):
            exp = self.daily_expenses[self.expense_idx]
            print(f"    [EXPENSE] {exp.name:20s} | {action.decision:6s} | "
                  f"spend={spend:>7.0f} | bal={self.state['balance']:>8.0f}")

        self.expense_idx += 1

        if self.expense_idx >= len(self.daily_expenses):
            # === END OF DAY ===

            # FIX 3: Savings = money NOT spent from today's allowance
            # If agent spent LESS than daily_allowance, the difference is saved toward goal
            actual_savings = max(0.0, self.daily_allowance - self.daily_spend_total)

            previous_goal = self.state["goal_remaining"]
            self.state["goal_remaining"] = max(0.0, self.state["goal_remaining"] - actual_savings)
            goal_progress = previous_goal - self.state["goal_remaining"]

            # Reward for goal progress (scaled)
            reward += goal_progress * 0.01

            # Bonus if goal is fully achieved
            if self.state["goal_remaining"] <= 0:
                reward += 50.0

            print(f"  [DAY END] day {self.state['days_left']:>2d} | "
                  f"spent={self.daily_spend_total:>7.0f} | allowance={self.daily_allowance:>7.0f} | "
                  f"saved={actual_savings:>7.0f} | goal_left={self.state['goal_remaining']:>8.0f} | "
                  f"bal={self.state['balance']:>8.0f}")

            # Reset for next day
            self.expense_idx = 0
            self.daily_spend_total = 0.0
            self.state["days_left"] -= 1

            # Recompute allowance for the new day
            self._recompute_daily_allowance()

            num_exps = self.expense_gen.rng.randint(2, 4)
            self.daily_expenses = self.expense_gen.generate_daily_expenses(num_exps)

        done = bool(self.state["days_left"] <= 0)

        # Episode-end sparse reward
        if done:
            if self.state["goal_remaining"] <= 0:
                reward += 50.0
            else:
                reward -= 30.0
            if self.state["stress_level"] > 0.7:
                reward -= 20.0

        return self._get_observation(), reward, done, {}

    def get_state(self) -> Dict[str, Any]:
        return self.state