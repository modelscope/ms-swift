# Copyright (c) ModelScope Contributors. All rights reserved.
from .causal_lm import CustomCrossEntropyLoss
from .decision import ClefLoss, JevDistillLoss, OmniJevLoss, ScoringLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
    # decision (typed-decision System-1 scoring)
    'scoring': ScoringLoss,
    'jev_distill': JevDistillLoss,
    'clef': ClefLoss,
    'omnijev': OmniJevLoss,
}
