import os

sequence_parallel = os.environ.get('SEQUENCE_PARALLEL_IMPL', 'ulysses')

if sequence_parallel == 'ulysses':
    from .ulysses import SequenceParallel
    sequence_parallel = SequenceParallel()
else:
    raise ValueError(f'Invalid sequence parallel implementation: {sequence_parallel}')
