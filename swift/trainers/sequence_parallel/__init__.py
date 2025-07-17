import os

sequence_parallel = os.environ.get('SEQUENCE_PARALLEL_IMPL', 'ulysses')

if sequence_parallel == 'xtuner':
    from .xtuner import XTuner
    sequence_parallel = XTuner()
elif sequence_parallel == 'ulysses':
    from .ulysses import Ulysses
    sequence_parallel = Ulysses()
elif sequence_parallel == 'ring_attention':
    from .ring_attention import RingAttention
    sequence_parallel = RingAttention()
else:
    raise ValueError(f'Invalid sequence parallel implementation: {sequence_parallel}')
