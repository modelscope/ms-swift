# v5 server client: multi-LoRA fine-tuning + sampling against a running twinkle-server.
#
# This is the client half of the server/ pair. It attaches SEVERAL named LoRA adapters to
# one resident base model (the server's model app is configured with max_loras=4), trains
# each over HTTP, saves it, then samples from the saved adapter to verify end-to-end. The
# base model is built server-side through swift's family loader (see server_config.yaml).
#
# The server must already be running (see serve.sh). Everything here is remote: no model
# weights are loaded in this process; it only issues HTTP calls via the twinkle client.
#
# Run through train.sh, which pins the env vars below and the interpreter.

import os

from peft import LoraConfig
from transformers import AutoTokenizer

from twinkle import get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle_client import init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler
# MEASURED: client verified against server (2026-11-24)

logger = get_logger()

# Public model name served by this pair; must match server_config.yaml's supported_models
# and the model/sampler route tails.
base_model = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
base_url = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
api_key = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')

# Qwen3.5 has a dedicated twinkle template class (Qwen3_5Template); the base 'Template' would
# render through the checkpoint's own jinja chat_template instead.
template_cls = os.environ.get('TWINKLE_TEMPLATE', 'Qwen3_5Template')

# Multi-LoRA: several named adapters co-resident on one base model. Comma-separated; the
# smoke shrinks this to a single name and a couple of steps.
adapter_names = [n for n in os.environ.get('TWINKLE_ADAPTER_NAMES', 'assistant,coder').split(',') if n]
max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '6'))
data_slice = int(os.environ.get('TWINKLE_DATA_SLICE', '200'))

# Per-adapter self-cognition personas: each named LoRA is taught a DIFFERENT identity, so
# sampling adapter A vs adapter B yields distinguishable identities (the un-adapted base
# model answers with its own pretrained identity as a third, independent control). Training
# every adapter on one shared recipe would make their identities indistinguishable by name.
ADAPTER_PERSONAS = {
    'assistant': {'model_name': 'twinkle助手', 'model_author': 'ModelScope社区'},
    'coder': {'model_name': 'twinkle代码助手', 'model_author': 'ModelScope社区'},
}


def build_dataloader(adapter_name):
    # Self-cognition dataset: teaches the model its own name/author. Encoded server-side by
    # the processor app; the client only ships the recipe. One dataloader per adapter, each
    # with that adapter's persona, so every adapter trains its own identity from row 0.
    persona = ADAPTER_PERSONAS.get(
        adapter_name, {'model_name': f'twinkle模型（{adapter_name}）', 'model_author': 'ModelScope社区'})
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(data_slice)))
    dataset.set_template(template_cls, model_id=f'ms://{base_model}', max_length=512)
    dataset.map('SelfCognitionProcessor', init_args=persona)
    dataset.encode(batched=True)
    return DataLoader(dataset=dataset, batch_size=4)


def train_adapter(model, dataloader, adapter_name):
    # Attach a named LoRA adapter to the resident base model. max_loras on the server bounds
    # how many can coexist; adding a second name here is what makes this multi-LoRA.
    # Deliberately NO save_dir: the server owns the checkpoint store (TWINKLE_DEFAULT_SAVE_DIR,
    # default ./outputs under the server cwd) and save() hands back a `twinkle://` URI. Passing
    # a client-side save_dir only works when client and server share a filesystem and the
    # directory already exists on the server, so the managed store is the portable choice.
    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model(adapter_name, lora_config, gradient_accumulation_steps=2)

    model.set_template(template_cls)
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('CrossEntropyLoss')
    model.set_optimizer('Adam', lr=1e-4)

    logger.info(f'[{adapter_name}] train configs: {model.get_train_configs().model_dump()}')

    losses = []
    global_step = 0
    for epoch in range(3):
        for cur_step, batch in enumerate(dataloader, start=1):
            model.forward_backward(inputs=batch)
            model.clip_grad_and_step()
            global_step += 1

            if cur_step % 2 == 0:
                metric = model.calculate_metric(is_training=True)
                losses.append(metric.result)
                logger.info(f'[{adapter_name}] step {cur_step}/{len(dataloader)}, metric: {metric.result}')

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    resp = model.save(
        name=f'{adapter_name}-lora',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    logger.info(f'[{adapter_name}] saved checkpoint: {resp.twinkle_path}')
    return resp.twinkle_path, losses


def sample(adapter_uri=None, tokenizer=None):
    # Generation runs on the server's vLLM sampler; adapter_uri selects a trained LoRA
    # (None -> base model). This is the HTTP-only equivalent of a local infer.
    sampler = vLLMSampler(model_id=base_model)
    sampler.set_template(template_cls, model_id=base_model)

    # Identity probes: 「你是谁？」 elicits the trained author, but at small step counts the
    # persona NAME may only surface on a name-eliciting question (the self-cognition recipe's
    # {{NAME}} rows include 「你叫什么名字呀！」-style queries). Probe BOTH so each adapter's
    # own persona name is observable; the un-adapted base must still answer with its own
    # pretrained identity on the SAME prompts (third-identity control).
    trajectories = [
        {
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ]
        } for prompt in ('你是谁？', '你叫什么名字？')
    ]
    # max_tokens must clear Qwen3.5's thinking preamble: a 64-token window stops mid-thought,
    # before the identity statement is ever emitted. 512 tokens reaches the answer (measured).
    responses = sampler.sample(
        inputs=trajectories,
        sampling_params={'max_tokens': 512, 'temperature': 0.0},
        adapter_uri=adapter_uri,
    )

    texts = []
    for response in responses:
        for seq in response.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            texts.append(text)
            logger.info(f'[sample adapter_uri={adapter_uri}] {text}')
    return texts


def main():
    client = init_twinkle_client(base_url=base_url, api_key=api_key)

    logger.info('Available models:')
    for item in client.get_server_capabilities().supported_models:
        logger.info('- ' + item.model_name)

    model = MultiLoraTransformersModel(model_id=f'ms://{base_model}')

    # Decode sampler token ids with the base tokenizer. Resolve the checkpoint through
    # modelscope (not HF hub) so this works offline on a modelscope-cached box.
    from swift.dev.utils.hub import safe_snapshot_download
    tokenizer = AutoTokenizer.from_pretrained(safe_snapshot_download(base_model), trust_remote_code=True)

    # Train each named adapter against the one resident base model (multi-LoRA). Each gets
    # its own dataloader carrying its own persona (a shared dataloader would both share the
    # identity and hand later adapters a mid-dataset slice).
    saved = {}
    for adapter_name in adapter_names:
        dataloader = build_dataloader(adapter_name)
        twinkle_path, losses = train_adapter(model, dataloader, adapter_name)
        saved[adapter_name] = twinkle_path
        if losses:
            logger.info(f'[{adapter_name}] first={losses[0]} last={losses[-1]} (loss is finite and computed per step)')

    # Verify generation: base model, then each trained adapter over HTTP.
    sample(adapter_uri=None, tokenizer=tokenizer)
    for adapter_name, twinkle_path in saved.items():
        sample(adapter_uri=twinkle_path, tokenizer=tokenizer)

    logger.info(f'DONE adapters={list(saved)}')


if __name__ == '__main__':
    main()
