from .causal_lm import CustomCrossEntropyLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import GenerativeRerankerLoss, ListwiseGenerativeRerankerLoss, ListwiseRerankerLoss, RerankerLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'reranker': RerankerLoss,
    'generative_reranker': GenerativeRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
    'listwise_generative_reranker': ListwiseGenerativeRerankerLoss,
}
