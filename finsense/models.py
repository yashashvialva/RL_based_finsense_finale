from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
 
 
class Expense(BaseModel):
    name: str = Field(..., description="Name of the expense (e.g., Swiggy, Uber)")
    category: str = Field(..., description="Category (food, transport, rent, etc.)")
    amount: float = Field(..., description="Amount in INR")
    necessity_tag: Literal["essential", "semi-essential", "discretionary"] = Field(..., description="Necessity level")
    context: Literal["normal", "weekend", "emergency"] = Field("normal", description="Context of the expense")
 
 
class ObservationModel(BaseModel):
    # Core financials — matches env.py state keys exactly
    balance: float = Field(..., description="Current balance in INR")
    goal_total: float = Field(..., description="Total savings goal in INR")
    goal_remaining: float = Field(..., description="Amount remaining to reach the savings goal")
    days_left: int = Field(..., description="Days left until the deadline")
    required_savings_per_day: float = Field(..., description="Savings needed per day to reach the goal")
    daily_allowance: float = Field(..., description="How much the agent can spend today and still hit the goal")
 
    # Risk & stress
    stress_level: float = Field(..., ge=0.0, le=1.0, description="Current stress level")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk level of financial ruin")
 
    # Spending history
    recent_spending: List[float] = Field(default_factory=list, description="Last 5 transaction amounts")
    avg_daily_spend: float = Field(..., description="Rolling average daily spend")
 
    # Fixed costs & shocks
    expected_fixed_expenses: float = Field(..., description="Expected mandatory fixed expenses")
    income_shock_active: bool = Field(default=False, description="Whether a negative income shock is active")
 
    # Current expense to decide on
    current_expense: Optional[Dict[str, Any]] = Field(None, description="The incoming expense to decide on")
 
    # Task context
    task_id: str = Field(..., description="Current task identifier")
    salary: float = Field(..., description="Initial salary / starting balance")

    # Multi-agent world events
    active_events: List[str] = Field(default_factory=list, description="Currently active macro events")
 
 
class ActionModel(BaseModel):
    decision: Literal["allow", "reduce", "avoid"] = Field(..., description="Action to take on the current expense")
    approved_amount: float = Field(..., description="Amount approved (0 if avoid, full/partial if allow/reduce)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the action")
 
 
class StateModel(BaseModel):
    current_day: int
    total_days: int
    balance: float
    initial_goal: float
    current_goal_remaining: float
    stress_level: float
    risk_level: Literal["low", "medium", "high"]
    seed: int
    task_id: str
    expected_fixed_expenses: float
    income_shock_active: bool
    recent_spending: List[float]
    user_type: Literal["impulsive", "balanced", "frugal"]
    current_expense_idx: int
    daily_expenses: List[Expense]
    daily_expense_idx: int
    terminated: bool
    truncated: bool
 
 
class RewardModel(BaseModel):
    value: float = Field(..., description="Total reward for this step")
    components: Dict[str, float] = Field(default_factory=dict, description="Breakdown of reward components")