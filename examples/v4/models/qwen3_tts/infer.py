import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    'output/Qwen3-TTS-12Hz-1.7B-Base/vx-xxx/checkpoint-xxx',
    device_map='cuda:0',
    dtype=torch.bfloat16,
)

wavs, sr = tts.generate_custom_voice(
    text='我不是俱乐部的会员。只是一大早被克洛琳德敲门叫醒，说有个不错的剧本可以体验，这才决定加入的。',
    speaker='speaker_test',
)
sf.write('output.wav', wavs[0], sr)
