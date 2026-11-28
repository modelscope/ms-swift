# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Union, cast

import json
from PIL import Image

from twinkle.template.contract import ConversationSample, KTOSample, PreferenceSample
from .runtime import get_logger
from .utils import Messages, Tool, get_last_user_round, messages_to_history, normalize_messages

logger = get_logger()


def _lift_leading_systems(messages: Messages) -> Optional[str]:
    """Pop every LEADING system message and join the blocks into one system string.

    Returns None when the conversation has no leading system (the caller's ``system`` key then
    wins). A mid-conversation system is left in place: the round contract only admits one at
    the head, and split_rounds fails loud on anything else.
    """
    if not messages or messages[0]["role"] != "system":
        return None
    blocks = []
    while messages and messages[0]["role"] == "system":
        blocks.append(messages.pop(0)["content"] or "")
    return "\n\n".join(block for block in blocks if block)


@dataclass
class StdTemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    # None: use default system; '': not use system
    system: Optional[str] = None
    tools: Optional[List[Tool]] = None

    label: Optional[int] = None
    channel: Optional[str] = None

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    objects: Dict[str, Any] = field(default_factory=dict)

    margin: Optional[float] = None  # for reward modeling
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)  # from dataset
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    mm_processor_kwargs: Dict[str, Any] = field(default_factory=dict)

    #: the messages arrived as content-parts lists and were folded down to text here. Some shipped
    #: jinjas branch on the shape (they scan for a query turn whose content *is a string*), so the
    #: renderer has to know which shape the caller used even though it renders text either way.
    content_parts: bool = False

    def __post_init__(self):
        self.image_idx = 0
        self.audio_idx = 0
        self.video_idx = 0
        self.ref_idx = 0
        self.bbox_idx = 0
        if self.images and not isinstance(self.images, (list, tuple)):
            self.images = [self.images]
        if self.videos and not isinstance(self.videos, (list, tuple)):
            self.videos = [self.videos]
        if self.audios and not isinstance(self.audios, (list, tuple)):
            self.audios = [self.audios]

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> "StdTemplateInputs":
        inputs = deepcopy(inputs)
        kwargs = {}
        for key in ["label", "channel", "margin"]:
            if key in inputs:
                kwargs[key] = inputs[key]
        messages = inputs["messages"]
        tools = inputs.get("tools")
        objects = inputs.get("objects") or {}
        chat_template_kwargs = inputs.get("chat_template_kwargs") or {}
        mm_processor_kwargs = inputs.get("mm_processor_kwargs") or {}

        messages = normalize_messages(messages)

        for message in messages:
            if message["role"] == "tool_response":
                message["role"] = "tool"
            if message["role"] in {"tool_call", "tool"} and not isinstance(message["content"], str):
                message["content"] = json.dumps(message["content"], ensure_ascii=False)

        content_parts = any(
            isinstance(message["content"], list) and message["content"]
            and not isinstance(message["content"][0], int) for message in messages)
        media_kwargs = StdTemplateInputs.remove_messages_media(messages)
        # the system message rides the same parts folding as every other message (a parts-form
        # system would otherwise reach the renderer as a list), then lifts out as `system`.
        # ALL leading system blocks fold into one: agents emit several instruction messages
        # (codex sends two developer items, each mapped to system), and the round contract is
        # one system at the head -- lifting only the first would leave the second to crash
        # split_rounds, and dropping it would silently change the trained prompt.
        system = _lift_leading_systems(messages)
        if system is None:
            system = inputs.get("system")
        # media arrives either inside the messages or as row columns, never both
        for k in list(media_kwargs.keys()):
            mm_data = media_kwargs[k]

            inputs_mm_data = inputs.get(k)
            if isinstance(inputs_mm_data, str):
                inputs_mm_data = [inputs_mm_data]
            inputs_mm_data = (inputs_mm_data or []).copy()
            if mm_data:
                assert not inputs_mm_data, f"self.{k}: {inputs_mm_data}"
            else:
                media_kwargs[k] = inputs_mm_data

        all_keys = {f.name for f in fields(StdTemplateInputs)}
        extra_kwargs = dict(inputs.get("extra_kwargs") or {})
        extra_kwargs.update({k: v for k, v in inputs.items() if k not in all_keys})
        return cls(
            messages=messages,
            system=system,
            tools=tools,
            objects=objects,
            chat_template_kwargs=chat_template_kwargs,
            extra_kwargs=extra_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
            content_parts=content_parts,
            **kwargs,
            **media_kwargs,
        )

    @staticmethod
    def remove_messages_media(messages: Messages) -> Dict[str, Any]:
        res = {"images": [], "audios": [], "videos": []}
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                # Native twinkle shape: the placeholder is already in the text and the media rides
                # per-message keys (images/videos/audios). Lift them so the renderer sees the media
                # and keeps the placeholder; otherwise the tag is stripped and the image is dropped
                # (the engine then holds 0 placeholders for N images).
                for key in ("images", "audios", "videos"):
                    value = message.get(key)
                    if value:
                        res[key].extend(value if isinstance(value, (list, tuple)) else [value])
            elif (isinstance(content, list) and content and isinstance(content[0], int)) or (
                isinstance(content, dict) and "token_ids" in content
            ):
                continue
            else:
                # a multimodal item list is rewritten to text here, the refs lifted into res
                message["content"] = StdTemplateInputs._lift_content_media(content, res)
        return res

    @staticmethod
    def _lift_content_media(content: list, res: Dict[str, Any]) -> str:
        new_content = ""
        for item in content:
            key: str = item["type"]
            value = item.get(key)
            if key == "text":
                new_content += value
                continue
            # media item (also the *_url form): placeholder stays in text, the ref is lifted
            if key.endswith("_url"):
                key = key[: -len("_url")]
            new_content += f"<{key}>"
            # twinkle's standard block carries the ref under 'url'/'path' rather than the type
            # key ({"type":"image","url":...}); fall back so the media is lifted, not dropped.
            if value is None:
                value = item.get("url") or item.get("path")
            if isinstance(value, dict):
                value = value["url"]
            if value:
                res[f"{key}s"].append(value)
        return new_content


# TODO: this pair is kept legacy-identical on purpose. Once legacy retires, rename to show the split at a
# glance: StdTemplateInputs -> TemplateInputs (one conversation), this class -> GroupedTemplateInputs (response group).
@dataclass
class TemplateInputs:
    chosen: StdTemplateInputs  # or Dict[str, Any]
    rejected: Optional[StdTemplateInputs] = None
    positive: List[StdTemplateInputs] = field(default_factory=list)  # or Dict[str, Any]
    negative: List[StdTemplateInputs] = field(default_factory=list)

    def __post_init__(self):
        all_keys = {f.name for f in fields(StdTemplateInputs)}
        for key in ["chosen", "rejected", "positive", "negative"]:
            value_dict = getattr(self, key, None)
            if not isinstance(value_dict, dict):
                continue
            if key in {"chosen", "rejected"}:
                setattr(self, key, StdTemplateInputs.from_dict(value_dict))
            else:
                res = []
                for i in range(len(value_dict["messages"])):
                    kwargs = {}
                    for k in all_keys:
                        val = value_dict.get(k)
                        if val is None:
                            continue
                        kwargs[k] = val[i]
                    res.append(StdTemplateInputs.from_dict(kwargs))
                setattr(self, key, res)

    @staticmethod
    def _compat_rejected_response(inputs: Dict[str, Any]):
        if "rejected_response" not in inputs:
            return
        messages = inputs["messages"]
        assert len(messages) > 0, f"messages: {messages}"
        idx = get_last_user_round(messages) + 1

        rejected_response = inputs.pop("rejected_response")
        if isinstance(rejected_response, str):
            rejected_responses = [{"role": "assistant", "content": rejected_response}]
        elif isinstance(rejected_response, list):
            rejected_responses = rejected_response
            for message in rejected_responses:
                if message["role"] == "user":
                    raise ValueError(
                        f"The 'user' role is not allowed in 'rejected_response' messages. Found: {message}"
                    )
        else:
            raise ValueError(f"rejected_response must be a str or list. rejected_response: {rejected_response}")
        # Check that the response is different from the rejected_response.
        if len(messages[idx:]) == 1 and len(rejected_responses) == 1:
            response = messages[idx]["content"]
            rejected_response = rejected_responses[0]["content"]
            assert rejected_response != response, f"rejected_response: {rejected_response}, response: {response}"
        inputs["rejected_messages"] = deepcopy(messages[:idx]) + rejected_responses

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> "TemplateInputs":
        return cls(**cls._split_inputs(inputs))

    @staticmethod
    def _split_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = deepcopy(inputs)

        has_rejected_messages = inputs.get("rejected_messages") is not None
        TemplateInputs._compat_rejected_response(inputs)
        kwargs = {}
        non_chosen_keys = ["rejected", "positive", "negative"]
        for prefix in ["chosen"] + non_chosen_keys:
            if prefix == "chosen":
                std_inputs = {
                    k: v for k, v in inputs.items() if not any(k.startswith(f"{p}_") for p in non_chosen_keys)
                }
            else:
                std_inputs = {k[len(f"{prefix}_") :]: v for k, v in inputs.items() if k.startswith(f"{prefix}_")}
            if std_inputs:
                kwargs[prefix] = std_inputs

        if not has_rejected_messages and kwargs.get("rejected") is not None:
            chosen = kwargs["chosen"]
            rejected = kwargs["rejected"]
            # Supplement additional key-value pairs
            for k, chosen_v in chosen.items():
                rejected_v = rejected.get(k)
                if chosen_v is not None and rejected_v is None:
                    rejected[k] = chosen_v

        return kwargs


# A raw row (dict) always encodes as one conversation -- its ``rejected_*`` columns are ignored, as in
# legacy SFT. Preference/KTO are selected only by their frozen runtime types.
SampleInput = Union[Dict[str, Any], Messages, StdTemplateInputs, TemplateInputs, PreferenceSample, KTOSample]


def normalize_sample(sample: SampleInput) -> Union[Dict[str, Any], TemplateInputs, PreferenceSample, KTOSample]:
    if isinstance(sample, (dict, TemplateInputs, PreferenceSample, KTOSample)):
        return sample
    if isinstance(sample, StdTemplateInputs):
        return TemplateInputs(chosen=sample)
    if isinstance(sample, list):
        return {"messages": sample}
    if is_dataclass(sample):
        return asdict(sample)
    raise TypeError(f"Cannot encode a {type(sample).__name__}; expected one of SampleInput.")


def preference_sample_from_row(row: Union[Dict[str, Any], TemplateInputs]) -> PreferenceSample:
    inputs = (
        {name: asdict(getattr(row, name)) for name in ('chosen', 'rejected') if getattr(row, name) is not None}
        if isinstance(row, TemplateInputs)
        else TemplateInputs._split_inputs(row)
    )
    if inputs.get('rejected') is None:
        raise ValueError("paired preference data requires a rejected conversation.")
    chosen = inputs['chosen']
    rejected = inputs['rejected']
    chosen = {**(chosen.get('extra_kwargs') or {}), **chosen}
    rejected = {**(rejected.get('extra_kwargs') or {}), **rejected}
    chosen.pop('extra_kwargs', None)
    rejected.pop('extra_kwargs', None)
    margin = chosen.pop("margin", None)
    rejected.pop("margin", None)
    return PreferenceSample(
        chosen=cast(ConversationSample, chosen),
        rejected=cast(ConversationSample, rejected),
        margin=margin,
    )


def kto_sample_from_row(row: Union[Dict[str, Any], TemplateInputs]) -> KTOSample:
    inputs = (
        {name: asdict(getattr(row, name)) for name in ('chosen', 'rejected') if getattr(row, name) is not None}
        if isinstance(row, TemplateInputs)
        else TemplateInputs._split_inputs(row)
    )
    conversation = inputs['chosen']
    if conversation.get('label') is None:
        raise ValueError("KTO data requires a boolean label.")
    conversation = {**(conversation.get('extra_kwargs') or {}), **conversation}
    conversation.pop('extra_kwargs', None)
    label = bool(conversation.pop("label"))
    rejected = inputs.get('rejected')
    if rejected is not None:
        rejected = {**(rejected.get('extra_kwargs') or {}), **rejected}
        rejected.pop('extra_kwargs', None)
    return KTOSample(conversation=cast(ConversationSample, conversation), label=label, rejected=rejected)
