#!/usr/bin/env python3
import os
import json
from openai import OpenAI

def test_mistral():
    # Load environment variables
    HF_TOKEN = os.getenv("HF_TOKEN","ollama")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
    MODEL_NAME = os.getenv("MODEL_NAME", "mistral:latest")

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    # Simple test prompt
    prompt = """You are a financial advisor in a reinforcement learning environment. Given the current state, decide on an action.

Current state:
- Balance: $30000
- Goal: Save $5000 in 15 days
- Daily allowance: $1667
- Required savings per day: $333
- Day: 1

Available actions:
- spend_high: Spend a lot (high risk, high reward)
- spend_medium: Spend moderately
- spend_low: Spend conservatively (low risk, low reward)
- save: Save money

Respond with ONLY a JSON object in this exact format:
{"action": "spend_low", "reasoning": "Brief explanation"}

Choose spend_low to be conservative and meet the savings goal."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        print(f"Raw response: {content}")

        # Try to parse JSON
        try:
            parsed = json.loads(content)
            print(f"Parsed JSON: {parsed}")
            print("SUCCESS: Mistral is working and returning valid JSON!")
            return True
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return False

    except Exception as e:
        print(f"API call failed: {e}")
        return False

if __name__ == "__main__":
    test_mistral()