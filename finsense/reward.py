from finsense.models import ActionModel

def calculate_reward(
    previous_goal_remaining: float,
    current_goal_remaining: float,
    balance: float,
    expected_fixed_expenses: float,
    days_left: int,
    action: ActionModel,
    stress_level: float,
    risk_level: str,
    recent_spending: list[float]
) -> tuple[float, float]:
    """
    Calculates the step reward.
    Returns: (reward, overspending_amount)
    """
    # Goal progress (positive if goal_remaining reduced)
    goal_progress = previous_goal_remaining - current_goal_remaining
    
    # Safe limit computation
    safe_limit = 0.0
    if days_left > 0:
        safe_limit = max(0.0, (balance - expected_fixed_expenses) / days_left)
        
    # Overspending penalty
    overspending_penalty = 0.0
    if action.approved_amount > safe_limit:
        overspending_penalty = (action.approved_amount - safe_limit) * 0.1
        
    # Stress penalty (dense)
    stress_penalty = 0.4 * stress_level
    
    # Risk penalty
    risk_map = {"low": 0.0, "medium": 1.0, "high": 5.0}
    risk_penalty = risk_map.get(risk_level, 0.0)
    
    # Inconsistency penalty (variance from moving average)
    inconsistency_penalty = 0.0
    if len(recent_spending) >= 2:
        avg_spending = sum(recent_spending) / len(recent_spending)
        inconsistency_penalty = abs(action.approved_amount - avg_spending) * 0.05
        
    # Final dense reward
    reward = goal_progress - overspending_penalty - stress_penalty - risk_penalty - inconsistency_penalty
    
    return reward, overspending_penalty
