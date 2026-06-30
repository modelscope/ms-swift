# This file is used to generate `audio_codes` in the dataset.
from modelscope import snapshot_download
from qwen_tts import Qwen3TTSTokenizer

from swift import load_dataset

BATCH_INFER_NUM = 32
dataset = load_dataset('qsdong/Qwen3-1.7-TTS-SFT-Furina')[0]
tokenizer_model_path = snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz')
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    tokenizer_model_path,
    device_map='cuda:0',
)
audio_codes = []

for i in range(0, len(dataset), BATCH_INFER_NUM):
    batch_lines = dataset[i:i + BATCH_INFER_NUM]
    batch_audios = [audios[0] for audios in batch_lines['audios']]
    enc_res = tokenizer.encode(batch_audios)
    audio_codes += [code.cpu().tolist() for code in enc_res.audio_codes]

dataset = dataset.add_column('audio_codes', audio_codes)
dataset.to_parquet('tts_data.parquet')
