---
title: FinSense RL
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FinSense RL Environment

FinSense is a goal-driven financial decision-making RL environment simulating Indian household budgeting. Given a savings goal (e.g., buying a phone worth Rs.20,000), a monthly salary, and unpredictable daily expenses, an AI agent learns to decide which expenses to **allow**, **reduce**, or **avoid** -- balancing stress, risk, and deadlines to hit the target in time.

Unlike toy environments, FinSense models a task that millions of Indians face every month: a fixed salary, unpredictable expenses, and a savings goal that keeps slipping. The environment captures delayed consequences, income shocks, trade-offs, and temporal pressure.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Features & Mechanics](#core-features--mechanics)
- [World Modeling Layer](#world-modeling-layer)
- [Multi-Agent System](#multi-agent-system)
- [Memory & Self-Improvement System](#memory--self-improvement-system)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Tasks & Difficulties](#tasks--difficulties)
- [Reward System](#reward-system)
- [Grading System](#grading-system)
- [Learning Evaluation Framework](#learning-evaluation-framework)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Architecture Overview

FinSense is built as a layered system where each component adds complexity and realism:

```
+--------------------------------------------------------------+
|                    INFERENCE LAYER                            |
|  inference.py (LLM Agent)  |  inference_local.py (Rule-Based)|
+--------------------------------------------------------------+
         |                              |
         v                              v
+--------------------------------------------------------------+
|                    MEMORY LAYER (SQLite)                      |
|  Store decisions -> Retrieve similar cases -> Bias decisions  |
|  Delayed commit with retroactive credit assignment            |
+--------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------+
|                    ENVIRONMENT CORE (env.py)                  |
|  State management | Reward calculation | Episode lifecycle   |
|  Context penalties | Stress mechanics | Bad decision tracking |
+--------------------------------------------------------------+
         |                    |                    |
         v                    v                    v
+------------------+  +----------------+  +------------------+
|  EVENT AGENT     |  | VENDOR AGENT   |  | EXPENSE GENERATOR|
|  Macro events    |  | Price adjust   |  | Daily expenses   |
|  (fuel, medical) |  | per category   |  | with context     |
+------------------+  +----------------+  +------------------+
```

---

## Core Features & Mechanics

### 1. Expense Categorization & Necessity Tags

Expenses are generated based on realistic Indian household economics and tagged by necessity: `essential`, `semi-essential`, and `discretionary`.

**Why it matters:** A robust financial policy should prune discretionary spending before cutting essentials.

| Necessity | Examples | Amount Range |
|-----------|----------|-------------|
| `essential` | House Rent, Groceries, Pharmacy, Electricity Bill | Rs.200 - Rs.25,000 |
| `semi-essential` | Swiggy/Zomato, Uber/Ola, Health Checkup | Rs.50 - Rs.3,000 |
| `discretionary` | Fine Dining, Flight Booking, Concert, Bar/Pub | Rs.200 - Rs.12,000 |

**Example expense generation:**
```python
# A single day might generate these 3 expenses:
Expense(name="Groceries",     category="food",          amount=930,  necessity_tag="essential",     context="normal")
Expense(name="Movie Tickets", category="entertainment", amount=450,  necessity_tag="discretionary", context="weekend")
Expense(name="Doctor Visit",  category="medical",       amount=1200, necessity_tag="essential",     context="emergency")
```

### 2. Tri-State Action Space

For every expense, the agent must output one of three decisions:

| Decision | Meaning | Example |
|----------|---------|---------|
| `allow` | Pay the full amount | "Groceries Rs.930" -> Pay Rs.930 |
| `reduce` | Pay a partial amount | "Uber Rs.400" -> Pay Rs.200 (take an auto instead) |
| `avoid` | Skip entirely | "Fine Dining Rs.3500" -> Pay Rs.0 |

**Why not binary?** Real life isn't yes/no. People negotiate, find alternatives, and partially fulfill needs. The tri-state space captures this nuance.

### 3. The Stress Mechanic (Anti-Exploit)

An accumulated `stress_level` (0.0 to 1.0) that increases when the agent avoids necessary expenses.

**Why it exists:** The biggest flaw in naive financial RL environments is the "avoid everything" exploit -- an agent that avoids all expenses trivially saves the most money. FinSense prevents this.

| Action | Necessity | Stress Impact |
|--------|-----------|---------------|
| `avoid` | `essential` | +0.30 |
| `avoid` | `semi-essential` | +0.15 |
| `avoid` | `discretionary` | +0.02 |
| `allow` | any | -0.05 (relief) |

**Penalty:** If stress exceeds 0.7 by the end of the episode, the agent suffers a massive -20.0 reward penalty. This forces the agent to actually pay for living essentials.

**Example walkthrough:**
```
Day 10: Agent avoids "Groceries" (essential)    -> stress: 0.00 -> 0.30
Day 10: Agent avoids "Pharmacy" (essential)     -> stress: 0.30 -> 0.60
Day 10: Agent avoids "Uber" (semi-essential)    -> stress: 0.60 -> 0.75  [DANGER!]
Day 10: Agent allows "Electricity" (essential)  -> stress: 0.75 -> 0.70
        End-of-episode penalty: -20.0 if stress > 0.70
```

### 4. Dynamic Daily Allowance

The environment automatically calculates and exposes a `daily_allowance` field:

```
daily_allowance = (balance - goal_remaining) / days_left
```

**Example:**
```
Balance: Rs.27,000 | Goal remaining: Rs.5,000 | Days left: 14
Spendable: Rs.22,000
Daily allowance: Rs.22,000 / 14 = Rs.1,571

If an expense comes in at Rs.3,000, the agent immediately knows it breaks the daily budget.
```

### 5. Stochastic Shock Events

Random, unpredictable financial events that alter the environment state between days (active on Medium/Hard tasks).

| Shock Type | Probability | Effect |
|------------|-------------|--------|
| `salary_delay` | 10% per day | Balance reduced by 5%, `income_shock_active` flag set |
| `emergency_expense` | 5% per day | Unavoidable Rs.2,500 "Emergency Medical Bill" injected |
| `discount` | 15% per day | 20% discount on the first expense of the next day |

**Example: Salary delay on Day 20 of the Hard task**
```
Before: Balance = Rs.80,000
[SHOCK] salary_delay | -4000
After:  Balance = Rs.76,000  (5% cut)
Agent must now recalculate its entire budget with Rs.4,000 less.
```

### 6. Delayed Consequences (Temporal Credit Assignment)

If the agent avoids an `essential` `medical` expense, there is a **50% probability** that a massive health emergency spawns 3 days later.

**Why probabilistic?** Deterministic consequences are too easy to exploit. The 50% chance mirrors real-world health uncertainties.

**Example walkthrough:**
```
Day 10: Agent avoids "Doctor Visit" (essential medical, Rs.600)
        System rolls dice: 0.35 < 0.50 -> Consequence scheduled for Day 7!
Day  9: Normal day. Agent saves money. Feels good.
Day  8: Normal day. No consequences yet.
Day  7: "Delayed Health Emergency" (Rs.3,000, essential) appears!
        The Rs.600 savings on Day 10 just cost Rs.3,000 on Day 7.
```

---

## World Modeling Layer

### Context-Aware Expenses

Every expense includes a `context` field reflecting the situation in which it arises:

| Context | Probability | Meaning |
|---------|-------------|---------|
| `normal` | 70% | Regular day, no special circumstances |
| `weekend` | 20% | Weekend spending pressure (lifestyle inflation) |
| `emergency` | 10% | Urgent/critical situation requiring immediate attention |

**Reward integration (how context affects the agent's score):**

| Situation | Penalty | Rationale |
|-----------|---------|-----------|
| Emergency + avoid essential/semi-essential | -10.0 scaling | Avoiding critical needs in emergencies is dangerous |
| Weekend + allow discretionary | -5.0 scaling | Weekend lifestyle spending should be controlled |

**Example:**
```
Expense: "Doctor Visit" | Rs.1,200 | essential | context=emergency
Agent decides: "avoid"
Result: -10.0 context penalty applied!
        (Skipping a doctor in an emergency is financially and medically catastrophic)

Expense: "Fine Dining" | Rs.3,500 | discretionary | context=weekend
Agent decides: "allow"
Result: -5.0 context penalty applied!
        (Weekend splurging on unnecessary dining drains the budget)
```

---

## Multi-Agent System

FinSense uses a multi-agent architecture where two specialized agents simulate market forces:

### EventAgent -- Macro Economic Events

The `EventAgent` triggers world-level events that affect the entire financial landscape. Events use **percentage-based windows** so they work across all task durations (15, 30, or 45 days).

| Event | Window | Probability | Intensity Range | Duration | Affected Categories |
|-------|--------|-------------|-----------------|----------|-------------------|
| `fuel_crisis` | 20-55% and 75-90% | 15% | 1.2x - 1.8x | ~10% of episode | transport |
| `inflation` | 30-70% | 12% | 1.1x - 1.5x | ~12% of episode | food, utility |
| `medical_surge` | Any time | 8% | 1.3x - 2.0x | ~8% of episode | medical |
| `festival_season` | 50-80% | 10% | 1.15x - 1.4x | ~10% of episode | food, entertainment |

**Example: Fuel crisis during Hard task (45 days)**
```
Day 35: [EVENT] >> fuel_crisis | intensity=1.52 | duration=4d
        All transport expenses are now 52% more expensive!

Day 35: "Metro Pass" normally Rs.1,500 -> now Rs.2,280
Day 34: "Uber/Ola" normally Rs.400 -> now Rs.608
Day 33: "Auto Rickshaw" normally Rs.150 -> now Rs.228
Day 32: Fuel crisis expires. Prices return to normal.
```

**Event stacking:** Multiple events can be active simultaneously, compounding their effects:
```
Day 22: [EVENT] inflation (1.4x food) + festival_season (1.25x food)
        Food multiplier = 1.4 * 1.25 = 1.75x
        "Groceries" normally Rs.800 -> now Rs.1,400!
```

### VendorAgent -- Dynamic Price Adjustments

The `VendorAgent` translates active events into price multipliers applied to each expense category. It recalculates multipliers every day based on currently active events.

```
[PRICES] Adjustments: {'food': 1.75, 'utility': 1.40, 'transport': 1.52}
```

This means:
- Food items cost 75% more than normal
- Utility bills cost 40% more than normal
- Transport costs 52% more than normal

---

## Memory & Self-Improvement System

The memory system is FinSense's most distinctive feature. It allows the agent to **learn from past episodes** by storing decisions in a persistent SQLite database and using them to bias future decisions.

### How It Works

```
Episode 1: Agent makes decisions -> Buffered in memory
           Episode endings (success/fail) -> Retroactive credit assignment
           Adjusted rewards committed to SQLite

Episode 2: Agent encounters similar expense
           Memory retrieves past cases for this (category, context, event)
           If confidence >= 0.65: Memory overrides rule-based decision
           New decisions buffered -> Committed at episode end
           
Episode N: Memory has thousands of entries
           Agent's decisions are heavily influenced by past successes/failures
           Performance measurably improves over episodes
```

### SQLite Schema

```sql
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    episode INTEGER,          -- Which episode this decision was made in
    day INTEGER,              -- Which day of the episode
    expense_type TEXT,        -- Category: food, transport, medical, etc.
    necessity TEXT,           -- essential, semi-essential, discretionary
    context TEXT,             -- normal, weekend, emergency
    event_type TEXT,          -- Active event: fuel_crisis, inflation, none
    price_multiplier REAL,    -- Current price multiplier from VendorAgent
    balance REAL,             -- Agent's balance at decision time
    days_left INTEGER,        -- Days remaining in episode
    action TEXT,              -- allow, reduce, avoid
    reward REAL,              -- Adjusted reward (retroactive credit assignment)
    outcome TEXT              -- "success" or "failed" (set at episode end)
);
```

### Retroactive Credit Assignment (The Key Innovation)

The memory system does **NOT** store rewards immediately. Instead, it buffers all decisions during an episode and only commits them after the episode ends, using the final outcome to adjust the stored rewards:

**If the episode SUCCEEDED (goal achieved):**
```python
# Reward smart discipline that led to success
if action == "avoid" and necessity in ["discretionary", "semi-essential"]:
    stored_reward = step_reward + 0.3   # Boost: "Avoiding non-essentials helped us succeed!"
else:
    stored_reward = step_reward         # Keep original
```

**If the episode FAILED (ran out of money):**
```python
# Severely penalize wasteful spending that contributed to failure
if action in ["allow", "reduce"] and amount > 0:
    if necessity == "discretionary":
        stored_reward = step_reward - 0.7   # Heavy penalty
    elif necessity == "semi-essential":
        stored_reward = step_reward - 0.4   # Moderate penalty
    else:  # essential
        stored_reward = step_reward         # No penalty for essentials!
else:
    stored_reward = step_reward             # "Avoid" actions are never penalized
```

**Why this matters:** Without retroactive credit assignment, the memory system falls into a trap called "Myopic Greed" -- it remembers that allowing a semi-essential expense gave a slightly higher immediate reward (due to stress relief), but doesn't connect that spending to the eventual episode failure. The delayed commit fixes this by tying every decision to the final outcome.

### Cascaded Memory Retrieval

When the agent needs advice from memory, it searches with progressive fallback:

```
Level 1: Exact match (expense_type + context + event_type)
         Example: "food" + "weekend" + "inflation" -> 8 matching cases

Level 2: Partial match (expense_type + context)
         Example: "food" + "weekend" -> 15 matching cases

Level 3: Category + necessity (expense_type + necessity)
         Example: "food" + "discretionary" -> 22 matching cases

Level 4: Broadest (expense_type only)
         Example: "food" -> 50 matching cases
```

### Confidence Scoring

The memory system calculates a confidence score before overriding the agent:

```python
confidence = (consistency * 0.5) + (volume * 0.2) + (reward_quality * 0.3)
```

| Factor | Weight | Description |
|--------|--------|-------------|
| Consistency | 50% | What fraction of matching cases agree on the same action? |
| Volume | 20% | How many cases support this action? (saturates at 8) |
| Reward Quality | 30% | What is the average reward of this action? |

**Override threshold:** The memory only overrides the agent's rule-based decision if `confidence >= 0.65`.

**Example:**
```
Expense: "Swiggy/Zomato" | semi-essential | weekend context | inflation event
Memory retrieves 12 matching cases:
  - 9 cases chose "reduce" with avg reward 0.72
  - 2 cases chose "allow" with avg reward 0.45
  - 1 case chose "avoid" with avg reward 0.61

Best action: "reduce" (9/12 = 75% consistency)
Confidence: (0.75 * 0.5) + (min(1.0, 9/8) * 0.2) + (0.72 * 0.3) = 0.375 + 0.2 + 0.216 = 0.791

Result: Memory override! Agent reduces the Swiggy order instead of its default heuristic.
```

---

## Observation Space

Each step returns a typed observation with these fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `balance` | float | Current account balance in INR | 45,000.0 |
| `goal_total` | float | Total savings goal in INR | 30,000.0 |
| `goal_remaining` | float | Amount still needed to reach goal | 15,000.0 |
| `days_left` | int | Days remaining in episode | 22 |
| `daily_allowance` | float | Max spend today to still hit goal | 681.0 |
| `required_savings_per_day` | float | Savings needed per day from now | 681.0 |
| `stress_level` | float [0-1] | Accumulated stress from avoiding essentials | 0.35 |
| `risk_level` | string | `low` / `medium` / `high` based on balance | "medium" |
| `expected_fixed_expenses` | float | Total fixed costs remaining | 40,000.0 |
| `income_shock_active` | bool | Whether a salary delay shock is active | false |
| `recent_spending` | list[float] | Last 5 transaction amounts | [930, 0, 450, 1200, 0] |
| `avg_daily_spend` | float | Rolling average daily spend | 1,250.0 |
| `salary` | float | Starting balance / salary | 80,000.0 |
| `task_id` | string | Current task identifier | "hard" |
| `current_expense` | dict | The incoming expense to decide on | (see below) |
| `active_events` | list[str] | Currently active macro events | ["fuel_crisis"] |

**current_expense structure:**
```json
{
  "name": "Swiggy/Zomato",
  "category": "food",
  "amount": 450.0,
  "necessity_tag": "semi-essential",
  "context": "weekend"
}
```

---

## Action Space

The agent returns one action per step:

```json
{
  "decision": "allow | reduce | avoid",
  "approved_amount": 0.0,
  "reasoning": "optional string"
}
```

**Constraints:**
- `decision` must be one of: `allow`, `reduce`, `avoid`
- For `allow`: `approved_amount` = full expense amount
- For `reduce`: `approved_amount` = partial amount (environment caps at balance)
- For `avoid`: `approved_amount` = 0.0

---

## Tasks & Difficulties

| Task | Goal | Days | Initial Balance | Shocks | User Type | Fixed Expenses |
|------|------|------|----------------|--------|-----------|---------------|
| **easy** | Save Rs.5,000 | 15 | Rs.30,000 | Off | balanced | Rs.12,000 |
| **medium** | Save Rs.15,000 | 30 | Rs.60,000 | On | balanced | Rs.25,000 |
| **hard** | Save Rs.30,000 | 45 | Rs.80,000 | On | impulsive | Rs.40,000 |

**Hard task breakdown:**
```
Starting balance:          Rs. 80,000
Savings goal:              Rs. 30,000
Available for spending:    Rs. 50,000
Fixed expenses expected:   Rs. 40,000
Discretionary budget:      Rs. 10,000 over 45 days = Rs.222/day

With shocks (salary delays, emergencies) and impulsive user type,
the margin of error is extremely thin. One bad day can derail the goal.
```

---

## Reward System

The reward is computed per-step and combines multiple signals:

```
base_reward = +10.0 (for any valid action)
- stress_penalty:     stress_level * -20.0 (punishes high stress)
- context_penalty:    -10.0 (avoiding emergencies) or -5.0 (weekend discretionary)
- savings_bonus:      +5.0 to +15.0 (for staying on track to hit the goal)
- overspend_penalty:  -5.0 to -10.0 (for exceeding daily allowance)
- endgame_bonus:      +30.0 (on final step if goal is reached)
- endgame_penalty:    -30.0 (on final step if goal is not reached)
```

The raw reward (range: roughly -100 to +100) is normalized to [0.01, 0.99]:
```python
normalized = (reward + 100.0) / 200.0
final = 0.01 + (normalized * 0.98)    # Strictly bounded
```

---

## Grading System

Each task has a separate deterministic grader returning a float in [0.01, 0.99]:

**Task 1 (Easy) -- Monthly Saver:**
```
score = goal_progress + stress_bonus
goal_progress = savings_completed / savings_goal        (0.0 - 0.99)
stress_bonus  = 0.2 * (1 - stress/0.7)                 (0.0 - 0.19)
```

**Task 2 (Medium) -- Quarter Goal:**
```
score = (0.6 * progress) + (0.2 * efficiency) + (0.2 * risk_score)
progress   = savings_completed / savings_goal
efficiency = 1.0 - (days_used / total_days)
risk_score = {low: 1.0, medium: 0.5, high: 0.0}
```

**Task 3 (Hard) -- Multi-Goal Chaos:**
```
score = (0.5 * savings) + (0.25 * stress_mgmt) + (0.25 * balance_health)
savings       = savings_completed / savings_goal
stress_mgmt   = 1.0 - stress_level
balance_health = balance / 15000  (emergency fund target)
```

---

## Learning Evaluation Framework

### experiment_runner.py

The experiment runner demonstrates that the memory system improves the agent's performance over episodes. It runs in three phases:

```
Phase 0: PRE-TRAINING (2N or 4N episodes)
  - Rule-based agent with 15% random exploration
  - Decisions committed to memory DB with retroactive credit assignment
  - Purpose: Seed the memory with diverse experiences (both good and bad)
  - Note: Medium task uses 4N episodes for better memory quality.

Phase 1: BASELINE (N episodes, NO memory)
  - Clean rule-based agent (no randomness, no memory overrides)
  - Establishes the performance ceiling of pure heuristics

Phase 2: EVALUATION (N episodes, WITH memory)
  - Same rule-based agent BUT with memory bias overrides
  - Uses the pre-trained memory DB from Phase 0
  - Memory overrides kick in when confidence >= 0.65
```

### Running the Evaluation

```bash
# Run 10 episodes on the Hard task
python experiment_runner.py --episodes 10 --task hard

# Run 5 episodes on the Easy task
python experiment_runner.py --episodes 5 --task easy
```

### Sample Output

```
======================================================================
  DETAILED RESULTS
======================================================================

 Episode | No Memory Reward | With Memory Reward | No Mem Bad | Mem Bad
---------+------------------+--------------------+------------+--------
       1 |            73.59 |              70.86 |          0 |       3
       2 |            73.78 |              76.95 |          0 |       2
       3 |            68.54 |              72.26 |          0 |       1
       ...
      10 |            65.87 |              71.09 |          0 |       1

======================================================================
  LEARNING ASSESSMENT
======================================================================
  Reward improvement:         +2.32
  [YES] Memory shows POSITIVE learning effect!
```

### Generated Plots

The runner generates a 4-panel matplotlib chart saved as `finsense_learning_evaluation_<task>.png`:

1. **Total Reward per Episode** -- Line chart comparing with/without memory
2. **Bad Decisions per Episode** -- Bar chart showing decision quality
3. **Graded Score per Episode** -- Line chart of official task scores
4. **Cumulative Reward** -- Area chart showing long-term reward accumulation

---

## Quick Start

### Standardized Quick Start (OpenEnv Compliant)

The environment is fully compliant with OpenEnv multi-mode deployment.

```bash
# 1. Install dependencies
uv sync

# 2. Run the environment server
uv run server

# 3. Verify validation
./scripts/validate-submission.sh https://<your-space>.hf.space
```

### Testing Locally without API Credits

Use the rule-based agent to verify all environment mechanics locally:

```bash
# Run all tasks (easy, medium, hard) without memory
python inference_local.py --no-memory

# Run all tasks with memory enabled
python inference_local.py --use-memory
```

### Running with an LLM

```bash
# Set environment variables
export API_BASE_URL="http://localhost:11434/v1/"
export MODEL_NAME="mistral:latest"
export HF_TOKEN="your-token"

# Run inference (auto-falls back to rule-based if LLM is unavailable)
python inference.py
```

### Running Learning Evaluation

```bash
# 10-episode comparison on Hard task with matplotlib plots
python experiment_runner.py --episodes 10 --task hard
```

### REST API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "avoid", "approved_amount": 0.0, "reasoning": "discretionary"}'

# Get current state
curl -X GET http://localhost:7860/state
```

---

## Project Structure

```
finsense-rl/
├── server/                        # Multi-mode server package
│   └── app.py                     # Main entry point
├── scripts/                       # Automation & validation
│   └── validate-submission.sh
├── finsense/                      # Core environment package
│   ├── env.py                     # FinSenseEnv class
│   │                              #   - Episode lifecycle (reset/step)
│   │                              #   - Reward calculation with context penalties
│   │                              #   - Stress mechanics & daily allowance
│   │                              #   - Delayed consequences
│   │                              #   - Episode memory buffer & retroactive commit
│   │                              #   - Bad decision tracking
│   ├── models.py                  # Pydantic schemas
│   │                              #   - Expense, ObservationModel, ActionModel
│   │                              #   - StateModel, RewardModel
│   ├── agents.py                  # Multi-agent world layer
│   │                              #   - EventAgent: macro events (fuel, inflation)
│   │                              #   - VendorAgent: dynamic price adjustments
│   ├── memory.py                  # SQLite memory system
│   │                              #   - store_decision / retrieve_similar_cases
│   │                              #   - get_memory_bias_with_confidence
│   │                              #   - Cascaded fallback retrieval
│   ├── expense_generator.py       # Seeded expense engine
│   │                              #   - Category-weighted random generation
│   │                              #   - Context generation (normal/weekend/emergency)
│   │                              #   - Shock event checking
│   ├── graders.py                 # Per-task evaluation metrics
│   │                              #   - grade_task1 (easy), grade_task2 (medium)
│   │                              #   - grade_task3 (hard), grade_episode (router)
│   ├── server.py                  # FastAPI REST endpoints
│   ├── tasks.py                   # Task configurations (Easy/Medium/Hard)
│   └── reward.py                  # Reward signal calculations
├── inference.py                   # LLM agent with memory + graceful fallback
├── inference_local.py             # Rule-based agent with memory override
├── experiment_runner.py           # Learning evaluation with plots
│                                  #   - Phase 0: Pre-training with exploration
│                                  #   - Phase 1: Baseline evaluation (no memory)
│                                  #   - Phase 2: Memory-augmented evaluation
│                                  #   - Matplotlib comparison plots
├── openenv.yaml                   # OpenEnv metadata
├── Dockerfile                     # HF Space container config
├── pyproject.toml                 # Build metadata & entry points
└── requirements.txt               # Python dependencies
```

---

## Dependencies

```
fastapi
uvicorn
pydantic
openai
httpx
openenv-core
matplotlib
numpy
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:11434/v1/` | LLM API endpoint |
| `MODEL_NAME` | `""` | Model identifier (empty = use rule-based fallback) |
| `HF_TOKEN` | `"ollama"` | HuggingFace / API key |
| `USE_MEMORY` | `"1"` | Enable memory system (`1`=on, `0`=off) |

---

## STDOUT Format (OpenEnv Compliant)

The inference scripts emit exactly three line types:

```
[START] task=<task_name> env=finsense-rl model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

**Example output:**
```
[START] task=easy env=finsense-rl model=rule-based-fallback
[STEP] step=1 action=reduce(800) reward=0.50 done=false error=null
[STEP] step=2 action=avoid(0) reward=0.45 done=false error=null
[STEP] step=3 action=allow(930) reward=0.50 done=false error=null
[END] success=true steps=48 score=0.92 rewards=0.50,0.45,0.50,...
```

---

## Learning & Improvement

The memory-augmented agent measurably outperforms the baseline — 
average reward improves from ~51.5 to ~52.1 on the Medium task, 
with memory maintaining a stable graded score of 0.80 vs baseline 
which drops to 0.26 by episode 20.
The memory-augmented agent retrieves similar past decisions before acting,
using retroactive credit assignment to weight memories by episode outcome.

See: `finsense_learning_evaluation_hard.png`

---

## Multi-Agent Demo Example

```
Day 8:  [EVENT] fuel_crisis triggered | intensity=1.52 | duration=4d
        Transport expenses are now 52% more expensive
[PRICE] Uber/Ola: Rs.400 -> Rs.608 (x1.52)
[MEMORY] 6 past cases for transport+fuel_crisis — 5/6 chose reduce, avg reward 0.61, confidence 0.72
[OVERRIDE] Memory overrides rule-based decision -> reduce
[DECISION] Day 8 | Uber/Ola | normal | action=reduce | reward=0.48
```

---

## Training

FinSense supports LLM fine-tuning via TRL GRPO. Grader scores (0.01–0.99)
serve directly as GRPO reward signals.

```bash
# Step 1: Seed memory
python experiment_runner.py --episodes 40 --task hard

# Step 2: Fine-tune (see train_grpo.ipynb)
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[finsense_grader],
)
trainer.train()
```

Memory DB context is injected into the LLM prompt at each step so the
fine-tuned model learns to use past episodes as evidence.