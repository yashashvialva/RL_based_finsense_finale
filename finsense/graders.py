from finsense.models import StateModel
 
 
def grade_task1(state: StateModel) -> float:
    """
    Task 1 — Easy: Monthly Saver
    Goal: Save Rs.5000 in 15 days
    Grader focuses on: goal completion + low stress
    Score range: 0.0 to 1.0
    """
    # Goal progress (0.0 to 1.0)
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    goal_score = min(1.0, goal_completed / max(1.0, state.initial_goal))
 
    # Stress bonus — reward staying calm (low stress = good)
    stress_bonus = max(0.0, 0.2 * (1.0 - state.stress_level / 0.7))
 
    final_score = min(1.0, round(goal_score + stress_bonus, 4))
    return max(0.0, final_score)
 
 
def grade_task2(state: StateModel) -> float:
    """
    Task 2 — Medium: Quarter Goal
    Goal: Save Rs.15000 in 30 days with income shocks
    Grader focuses on: goal progress + efficiency + overspend control
    Score range: 0.0 to 1.0
    """
    # Goal progress
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    progress = min(1.0, goal_completed / max(1.0, state.initial_goal))
 
    # Efficiency: penalise agents that only save in the last few days
    # current_day counts up; total_days is max. Earlier completion = better.
    days_used_ratio = state.current_day / max(1, state.total_days)
    efficiency = max(0.0, 1.0 - days_used_ratio)  # higher if finished early
 
    # Risk control: penalise dangerous balance levels
    risk_score = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(state.risk_level, 0.5)
 
    final_score = round(
        (0.6 * progress) +
        (0.2 * efficiency) +
        (0.2 * risk_score),
        4
    )
    return min(1.0, max(0.0, final_score))
 
 
def grade_task3(state: StateModel) -> float:
    """
    Task 3 — Hard: Multi-Goal Chaos
    Goal: Save Rs.30000 + keep stress low + keep balance healthy
    Grader focuses on: all three simultaneously — hardest to get 1.0
    Score range: 0.0 to 1.0
    """
    # Primary: savings goal
    goal_completed = max(0.0, state.initial_goal - state.current_goal_remaining)
    savings_score = min(1.0, goal_completed / max(1.0, state.initial_goal))
 
    # Secondary: stress control (hard task has more shocks, stress harder to control)
    stress_score = max(0.0, 1.0 - state.stress_level)  # stricter than task1
 
    # Tertiary: balance health (must keep emergency fund)
    # Treat balance > 15000 as healthy
    emergency_fund_target = 15000.0
    balance_score = min(1.0, state.balance / emergency_fund_target) if hasattr(state, 'balance') else 0.5
 
    final_score = round(
        (0.5 * savings_score) +
        (0.25 * stress_score) +
        (0.25 * balance_score),
        4
    )
    return min(1.0, max(0.0, final_score))
 
 
def grade_episode(state: StateModel) -> float:
    """
    Router: calls the correct per-task grader based on state.task_id.
    Falls back to task1 grader if task_id is unrecognised.
    """
    graders = {
        "easy": grade_task1,
        "medium": grade_task2,
        "hard": grade_task3,
    }
    grader = graders.get(state.task_id, grade_task1)
    return grader(state)
 