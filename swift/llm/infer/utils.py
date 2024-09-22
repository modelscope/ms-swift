

class TokenListIteratorStreamer(BaseStreamer):

    def __init__(self, timeout: Optional[float] = None):
        self.token_queue = Queue()  # Queue[int]
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value: torch.Tensor) -> None:
        if value.ndim > 1:
            value = value[0]
        value = value.tolist()
        self.token_queue.put(value)

    def end(self) -> None:
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self) -> List[int]:
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

def _prepare_inputs(model: PreTrainedModel,
                    template: Template,
                    query: str,
                    history: History,
                    system: Optional[str] = None,
                    images: Optional[List[str]] = None,
                    *,
                    generation_config: GenerationConfig,
                    generation_info: Dict[str, Any],
                    stop_words: Optional[StopWords] = None,
                    adapter_names: Optional[List[str]] = None,
                    **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]:
    if stop_words is None:
        stop_words = []

    example = {
        'query': query,
        'history': history,
        'system': system,
        'images': images or [],  # for vl. str.
        'audios': kwargs.pop('audios', None) or [],
        'videos': kwargs.pop('videos', None) or [],
        'tools': kwargs.pop('tools', None),
        'objects': kwargs.pop('objects', None),
    }
    template.model = model
    inputs, tokenizer_kwargs = template.encode(example)

    truncation_strategy = kwargs.pop('truncation_strategy', 'delete')
    if len(inputs) == 0 and truncation_strategy == 'delete':
        # input_ids exceeds `max_length`. Please increase the value of `max_length`.
        return {}, tokenizer_kwargs, 0, example

    inputs.pop('labels', None)
    tokenizer = template.tokenizer
    device = next(model.parameters()).device
    if 'input_ids' in inputs:  # 1d
        input_ids = torch.tensor(inputs['input_ids'])[None]
        inputs['input_ids'] = input_ids
        token_len = input_ids.shape[1]
    if 'inputs_embeds' in inputs:  # 2d
        inputs_embeds = inputs['inputs_embeds'][None]
        inputs['inputs_embeds'] = inputs_embeds
        token_len = inputs_embeds.shape[1]

    inputs['attention_mask'] = torch.ones(token_len, dtype=torch.int64)[None]
    if 'token_type_ids' in inputs:
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])[None]
    model.eval()
    if not generation_config.do_sample:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        generation_config.bos_token_id = tokenizer.bos_token_id
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = 20  # fix max_length, max_new_tokens warning
        max_length = get_max_model_len(model.config)
        if max_length and token_len + generation_config.max_new_tokens > max_length:
            generation_config.max_new_tokens = max_length - token_len
            if generation_config.max_new_tokens <= 0:
                raise AssertionError(f'Current sentence length exceeds the model max_length: {max_length}')
    if template.suffix[-1] not in stop_words:
        stop_words.append(template.suffix[-1])
    inputs = to_device(inputs, device)
    if 'inputs_embeds' in inputs:
        inputs.pop('input_ids', None)
    if adapter_names is not None:
        inputs['adapter_names'] = adapter_names

    stopping_criteria = StoppingCriteriaList([StopWordsCriteria(tokenizer, stop_words, **tokenizer_kwargs)])
    inputs['stopping_criteria'] = stopping_criteria
    generation_info['num_prompt_tokens'] = token_len
    return inputs, tokenizer_kwargs, token_len, example


@torch.inference_mode()
def inference_stream(model: PreTrainedModel,
                     template: Template,
                     query: str,
                     history: Optional[History] = None,
                     system: Optional[str] = None,
                     images: Optional[List[str]] = None,
                     *,
                     generation_config: Optional[GenerationConfig] = None,
                     stop_words: Optional[StopWords] = None,
                     generation_info: Optional[Dict[str, Any]] = None,
                     adapter_names: Optional[List[str]] = None,
                     **kwargs) -> Iterator[Union[Tuple[str, History], Dict[str, Any]]]:
    """
    generation_config: Priority: generation_config > model.generation_config.
    """
    start_runtime = time.perf_counter()
    if history is None:
        history = []
    else:
        history = deepcopy(history)
    if generation_config is None:
        generation_config = getattr(model, 'generation_config')
    generation_config = deepcopy(generation_config)
    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()
    inputs, tokenizer_kwargs, token_len, example = _prepare_inputs(
        model,
        template,
        query,
        history,
        system,
        images,
        generation_config=generation_config,
        generation_info=generation_info,
        stop_words=stop_words,
        adapter_names=adapter_names,
        **kwargs)
    if len(inputs) == 0:
        return '', history

    # agent support
    is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
    if is_observation:
        history[-1][-1] = history[-1][-1] + query
        act_length = len(history[-1][-1])
        query = None

    if generation_config.num_beams != 1:
        error_msg = 'Streaming generation does not support beam search.'
        raise ValueError(error_msg)

    streamer = TokenListIteratorStreamer()
    return_dict = generation_config.return_dict_in_generate
    generation_kwargs = {'streamer': streamer, 'generation_config': generation_config, **inputs}
    result_queue = Queue()

    def _model_generate(*args, **kwargs):
        if is_torch_npu_available():
            torch.npu.set_device(model.device)
        res = model.generate(*args, **kwargs)
        result_queue.put(res)
        return res

    thread = Thread(target=_model_generate, kwargs=generation_kwargs)
    thread.start()
    raw_generate_ids, generate_ids = [], []

    if not is_observation:
        history.append(None)  # dummy

    print_idx = [0]
    first_num_space = [-1]

    is_finished = False
    while not is_finished:
        try:
            token_list = next(streamer)
            raw_generate_ids += token_list
        except StopIteration:
            is_finished = True
        res = {}
        generate_ids = template.get_generate_ids(torch.tensor(raw_generate_ids)[None], token_len)
        if return_dict and is_finished:
            thread.join()
            res = dict(result_queue.get())
            res['sequences'] = generate_ids
        generation_info['num_generated_tokens'] = len(generate_ids)
        response = template.generate_ids_to_response(
            generate_ids,
            is_finished,
            tokenizer_kwargs=tokenizer_kwargs,
            print_idx=print_idx,
            first_num_space=first_num_space)
        if not is_observation:
            history[-1] = [query, response]
        else:
            history[-1][-1] = history[-1][-1][:act_length] + response

        runtime = time.perf_counter() - start_runtime
        generation_info['runtime'] = runtime
        generation_info['samples/s'] = 1 / runtime
        generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
        if return_dict:
            res.update({'response': response, 'history': history})
            yield res
        else:
            yield response, history


@torch.inference_mode()
def inference(model: PreTrainedModel,
              template: Template,
              query: str,
              history: Optional[History] = None,
              system: Optional[str] = None,
              images: Optional[List[str]] = None,
              *,
              generation_config: Optional[GenerationConfig] = None,
              stop_words: Optional[StopWords] = None,
              generation_info: Optional[Dict[str, Any]] = None,
              stream: bool = False,
              verbose: bool = False,
              adapter_names: Optional[List[str]] = None,
              prompt_prefix: str = '[PROMPT]',
              output_prefix: str = '[OUTPUT]',
              **kwargs) -> Union[Tuple[str, History], Dict[str, Any]]:
    """
    generation_config: Priority: generation_config > model.generation_config.
    """
    runtime = time.perf_counter()
    if history is None:
        history = []
    else:
        history = deepcopy(history)
    if generation_config is None:
        generation_config = getattr(model, 'generation_config')
    generation_config = deepcopy(generation_config)
    if generation_info is None:
        generation_info = {}
    else:
        generation_info.clear()
    inputs, tokenizer_kwargs, token_len, example = _prepare_inputs(
        model,
        template,
        query,
        history,
        system,
        images,
        generation_config=generation_config,
        generation_info=generation_info,
        stop_words=stop_words,
        adapter_names=adapter_names,
        **kwargs)
    if len(inputs) == 0:
        return '', history

    # agent support
    is_observation = history[-1][-1].endswith('Observation:') if history and history[-1][-1] else False
    if is_observation:
        history[-1][-1] = history[-1][-1] + query
        query = None

    if stream and not verbose:
        logger.warning('Please set verbose to True to support TextStreamer, or use `inference_stream.`')
        stream = False
    streamer = None
    tokenizer = template.tokenizer
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    if verbose:
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            print(
                f'{prompt_prefix}{safe_tokenizer_decode(tokenizer, input_ids[0], **tokenizer_kwargs)}{output_prefix}',
                end='')
        else:
            print(f'[QUERY]{query}\n{output_prefix}', end='')

    return_dict = generation_config.return_dict_in_generate
    generate_ids = model.generate(streamer=streamer, generation_config=generation_config, **inputs)
    if return_dict:
        res = dict(generate_ids)
        generate_ids = generate_ids['sequences']
    generate_ids = template.get_generate_ids(generate_ids, token_len)
    generation_info['num_generated_tokens'] = len(generate_ids)
    if verbose and stream is False:
        response = tokenizer.decode(generate_ids, **tokenizer_kwargs)
        print(response)
    response = template.generate_ids_to_response(generate_ids, tokenizer_kwargs=tokenizer_kwargs)
    response = template.post_process_generate_response(response=response, example=example)
    if not is_observation:
        history.append([query, response])
    else:
        history[-1][-1] = history[-1][-1] + response
    runtime = time.perf_counter() - runtime
    generation_info['runtime'] = runtime
    generation_info['samples/s'] = 1 / runtime
    generation_info['tokens/s'] = generation_info['num_generated_tokens'] / runtime
    if return_dict:
        res['sequences'] = generate_ids
        res.update({'response': response, 'history': history})
        return res
    else:
        return response, history


def messages_join_observation(messages: Messages):
    """
        Joins observations from 'tool' message into the 'assistant' response.

        Example:
        ---------
        Original messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                  [{"location": "Hangzhou"}]\nObservations:'},
            {'role': 'tool', 'content': 'It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]

        Transformed messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                  [{"location": "Hangzhou"}]\nObservations: It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]
        """

    if len(messages) >= 2 and messages[-2]['role'] == 'assistant' and messages[-2]['content'] and messages[-2][
            'content'].endswith('Observation:'):
        assert messages[-1]['role'] == 'tool'
        observations = messages[-1]['content']
        messages.pop(-1)
        messages[-1]['content'] += observations
    return