from .acc import AccMetrics
from .embedding import InfonceMetrics, PairedMetrics
from .nlg import NlgMetrics
from .reranker import RerankerMetrics

# Add your own metric calculation method here, use `--metric xxx` to train
# The metric here will only be called during validation

metrics_map = {
    'acc': AccMetrics,
    'nlg': NlgMetrics,
    # embedding
    'infonce': InfonceMetrics,
    'paired': PairedMetrics,
    # reranker
    'reranker': RerankerMetrics,
}
