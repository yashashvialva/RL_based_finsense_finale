from openai import OpenAI

client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1/')
try:
    response = client.chat.completions.create(
        model='deepseek-r1:1.5b',
        messages=[{'role': 'user', 'content': 'Reply with only: {"test": "ok"}'}],
        max_tokens=50
    )
    raw = response.choices[0].message.content.strip()
    print('Raw response:')
    print(repr(raw))
    print('Content:')
    print(raw)
except Exception as e:
    print('Error:', e)