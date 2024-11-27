from swift.llm.template.agent.loss_scale import LossScale, loss_scale_map
from swift.llm.template.utils import ContextType


class TrainAllLossScale(LossScale):

    def get_loss_scale(self, context: str, context_type: ContextType, *args, **kwargs):
        return [context], [1.]


loss_scale_map['all'] = TrainAllLossScale()
