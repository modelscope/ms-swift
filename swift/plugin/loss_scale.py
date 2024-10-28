from swift.llm.template.agent.loss_scale import loss_scale_map


def custom_loss_scale(query: str, response: str):
    return [response], [1.0]


loss_scale_map['custom'] = custom_loss_scale
