from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = '浙江的省会在哪里?'
messages = [{'role': 'user', 'content': query}]
resp = client.chat.completions.create(model=model_type, messages=messages, seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# streaming
messages.append({'role': 'assistant', 'content': response})
query = '这有什么好吃的?'
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(model=model_type, messages=messages, stream=True, seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
