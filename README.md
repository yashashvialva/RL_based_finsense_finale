# FinSense RL Environment

A goal-driven financial decision-making RL environment simulating Indian household budgeting. 
This is an OpenEnv-compliant environment developed for the Meta x Hugging Face x Scaler Hackathon.

## Description
FinSense puts the AI agent in charge of managing an abstract Indian household budget. The agent receives a stream of daily expenses and must decide to either:
- **allow**: Full expense is deducted from the balance.
- **reduce**: A partial expense amount is paid.
- **avoid**: No deduction, but increases stress (especially for `essential` items).

The objective is to maximize long-term savings to meet specific financial goals while balancing stress, risk of ruin, tight deadlines, and stochastic financial shocks (salary delays, emergencies, etc.).

## Core Features
1. **Realistic Financial Features**: Takes into account necessity tags, moving averages of spendings, user behavioural patterns, and fixed expenses.
2. **Dense Rewards**: Integrates goal progression, overspending penalties, stress penalties, and inconsistency penalties to shape behavior continuously.
3. **Anti-Exploit Mechanics**: Avoiding essential and semi-essential expenses builds up stress. High stress results in heavy episode-end penalties to prevent the agent from just avoiding all expenses.
4. **Stochastic Elements**: Seed-controlled randomness injects salary delays, emergencies, and utility discounts.
5. **OpenEnv Ready**: Exposes 3 task difficulties (`easy`, `medium`, `hard`) alongside a FastAPI environment server.

## Installation

You can run the environment natively using Docker.

```bash
docker build -t finsense-env .
docker run -p 7860:7860 finsense-env
```

## Running Inference

The baseline OpenEnv inference script utilizes the OpenAI standard client. Make sure to export your credentials before running the agent.

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-hf-token-here"

python inference.py
```
