from openai import OpenAI
import os

HF_TOKEN = os.getenv('HF_TOKEN','ollama')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:11434/v1/')
MODEL_NAME = 'deepseek-r1:1.5b'
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# Test with the exact same prompt format as inference.py
test_prompt = """Financial agent. Decide on this expense.

Balance: 30000 | Goal Left: 5000 | Days: 15
Expense: Maintenance Fee | Amount: 1490 | Type: essential

Rules:
- essential -> allow full amount
- semi-essential -> reduce by 50%
- discretionary -> avoid

Reply ONLY with JSON: {"decision": "allow/reduce/avoid", "approved_amount": 0.0, "reasoning": "short"}"""

print("Sending prompt to DeepSeek:")
print(test_prompt)
print("\n" + "="*50)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
            {"role": "user", "content": test_prompt}
        ],
        max_tokens=80,
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    print("Raw response from DeepSeek:")
    print(repr(raw))
    print("\nResponse content:")
    print(raw)

    # Try to parse as JSON
    import json
    try:
        parsed = json.loads(raw)
        print("\nJSON parsed successfully:", parsed)
    except json.JSONDecodeError as je:
        print(f"\nJSON parse error: {je}")
        print("This is why the inference is failing!")

except Exception as e:
    print('Error:', e)