from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('baby.mp4', 'rb') as f:
#     vid_base64 = base64.b64encode(f.read()).decode('utf-8')
# video_url = f'data:video/mp4;base64,{vid_base64}'

# use local_path
# from swift.llm import convert_to_base64
# video_url = convert_to_base64(images=['baby.mp4'])['images'][0]
# video_url = f'data:video/mp4;base64,{video_url}'

# use url
video_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'

query = '描述这段视频'
messages = [{
    'role': 'user',
    'content': [
        {
            'type': 'video_url',
            'video_url': {
                'url': video_url
            }
        },
        {
            'type': 'text',
            'text': query
        },
    ]
}]
resp = client.chat.completions.create(model=model_type, messages=messages, temperature=0)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# streaming
query = '图中有几只羊'
image_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
messages = [{
    'role': 'user',
    'content': [
        {
            'type': 'image_url',
            'image_url': {
                'url': image_url
            }
        },
        {
            'type': 'text',
            'text': query
        },
    ]
}]
stream_resp = client.chat.completions.create(model=model_type, messages=messages, stream=True, temperature=0)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
