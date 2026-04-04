---
title: FinSense RL
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FinSense RL Environment

A goal-driven financial decision-making RL environment simulating Indian household budgeting.

## Overview

FinSense simulates real-world Indian household financial decision-making. An AI agent manages daily expenses across a fixed-horizon episode, deciding to **allow**, **reduce**, or **avoid** each incoming expense — balancing a savings goal, stress accumulation, and stochastic income shocks.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take an action, returns obs, reward, done, info |
| `/state` | GET | Get current state without advancing |
| `/tasks` | GET | List available tasks |
| `/health` | GET | Health check |

## Tasks

| Task | Goal | Days | Difficulty |
|------|------|------|------------|
| easy | Save ₹5,000 | 15 | Easy |
| medium | Save ₹15,000 | 30 | Medium |
| hard | Save ₹30,000 | 45 | Hard |

## Action Space

```json
{
  "decision": "allow | reduce | avoid",
  "approved_amount": 0.0,
  "reasoning": "optional string"
}
```

## Quick Start

```bash
# Reset environment
curl -X POST https://yashashvialva-finsense-rl.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST https://yashashvialva-finsense-rl.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "avoid", "approved_amount": 0.0, "reasoning": "discretionary"}'
```

## Setup

```bash
pip install -r requirements.txt
uvicorn finsense.server:app --host 0.0.0.0 --port 7860
```