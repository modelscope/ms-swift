import os

if os.environ.get('SEQUENCE_PARALLEL_IMPL', 'ulysses') == 'xtuner':
    from .xtuner import XTuner
    sequence_parallel = XTuner()
else:
    from .ulysses import Ulysses
    sequence_parallel = Ulysses()
