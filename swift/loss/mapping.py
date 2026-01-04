from .causal_lm import CrossEntropyLoss

loss_map = {
    'cross_entropy': CrossEntropyLoss,  # examples
    # embedding
    # 'cosine_similarity': cosine_similarity_func,
    # 'contrastive': contrastive_loss,
    # 'online_contrastive': online_contrastive_loss,
    # 'infonce': infonce_loss,
    # # reranker
    # 'reranker': reranker_loss,
    # 'generative_reranker': generative_reranker_loss,
    # 'listwise_reranker': listwise_reranker_loss,
    # 'listwise_generative_reranker': listwise_generative_reranker_loss,
}
