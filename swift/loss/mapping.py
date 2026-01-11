from .causal_lm import CustomCrossEntropyLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import ListwiseRerankerLoss, RerankerLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': RerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
