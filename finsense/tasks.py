from pydantic import BaseModel
from typing import Literal

class TaskConfig(BaseModel):
    task_id: str
    goal: float
    days: int
    allow_shocks: bool
    user_type: Literal["impulsive", "balanced", "frugal"]
    initial_balance: float
    expected_fixed_expenses: float

EASY_TASK = TaskConfig(
    task_id="easy",
    goal=5000.0,
    days=15,
    allow_shocks=False,
    user_type="balanced",
    initial_balance=30000.0,
    expected_fixed_expenses=12000.0
)

MEDIUM_TASK = TaskConfig(
    task_id="medium",
    goal=15000.0,
    days=30,
    allow_shocks=True,
    user_type="balanced",
    initial_balance=60000.0,
    expected_fixed_expenses=25000.0
)

HARD_TASK = TaskConfig(
    task_id="hard",
    goal=30000.0,
    days=45,
    allow_shocks=True,
    user_type="impulsive",
    initial_balance=80000.0,
    expected_fixed_expenses=40000.0
)

TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK
}
