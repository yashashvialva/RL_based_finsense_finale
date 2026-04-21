from openai import OpenAI
import os

HF_TOKEN = os.getenv('HF_TOKEN','ollama')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:11434/v1/')
MODEL_NAME = 'deepseek-r1:1.5b'
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

test_prompt = '''Financial agent. Decide on this expense.

Balance: 30000 | Goal Left: 5000 | Days: 15
Expense: Maintenance Fee | Amount: 1490 | Type: essential

Reply ONLY with JSON: {"decision": "allow", "approved_amount": 1490, "reasoning": "essential"}'''

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'system', 'content': 'You are a financial agent. Analyze expenses and respond ONLY with a valid JSON object in this exact format: {"decision": "allow", "approved_amount": 1490, "reasoning": "brief explanation"}. No other text, no markdown, no thinking traces.'},
                 {'role': 'user', 'content': test_prompt}],
        max_tokens=100,
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    print('Raw response:')
    print(repr(raw))
    print('Response content:')
    print(raw)

except Exception as e:
    print('Error:', e)