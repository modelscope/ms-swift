import io
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from openai import OpenAI


class LLMChat:
    def __init__(self, model, base_url, api_key='EMPTY', max_retries=3, **kwargs):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries
        )
        self.model = model
        self.kwargs = kwargs

    def chat(self, prompt, return_content_only=False, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            **(self.kwargs | kwargs)
        )
        return response.choices[0].message.content if return_content_only else response

    def vlchat(self, images, prompt, return_content_only=False, **kwargs):
        if not isinstance(images, (tuple, list)):
            images = [images]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {'url': self.img2b64(x)}
                        } for x in images
                    ] + [
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ],
            **(self.kwargs | kwargs)
        )
        return response.choices[0].message.content if return_content_only else response

    @staticmethod
    def img2b64(x, format='PNG', with_mime=True):
        if isinstance(x, (str, Path)):
            image = Image.open(x).convert('RGB')
        elif isinstance(x, np.ndarray):
            image = Image.fromarray(x).convert('RGB')
        elif isinstance(x, Image.Image):
            image = x.convert('RGB')
        else:
            raise TypeError(f'Unsupported input type: {type(x)}')

        imbin = io.BytesIO()
        image.save(imbin, format)
        image = base64.b64encode(imbin.getvalue()).decode('utf-8')

        if with_mime:
            return f'data:image/{format.lower()};base64,{image}'

        return image
