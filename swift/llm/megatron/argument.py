from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MegatronArguments:
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads = Optional[int] = None
    max_position_embeddings = Optional[int] = None
    

