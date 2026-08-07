from typing import TYPE_CHECKING
import os
import torch

if TYPE_CHECKING:
    from .base import Trainer, TrainingArguments

from swift import TrainerCallback, get_logger

logger = get_logger()


class MemorySnapshotCallback(TrainerCallback):
    """
    Record CUDA memory history and dump snapshot with specified interval steps.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.dump_interval = args.mem_snapshot_interval
        self.dump_path = args.mem_snapshot_path
        self._recording = False

    def _dump_and_visualize(self, step: int, tag: str = ''):
        rank = int(os.environ.get("RANK", 0))
        raw = f'snapshot_step{step}_rank{rank}'
        pickle_path = os.path.join(self.dump_path, f'{raw}.pickle')
        html_path = os.path.join(self.dump_path, f'{raw}.html')

        snapshot = torch.cuda.memory._snapshot()
        os.makedirs(os.path.dirname(os.path.abspath(pickle_path)), exist_ok=True)
        torch.cuda.memory._dump_snapshot(pickle_path)
        logger.info(f"{tag}CUDA memory snapshot dumped: {pickle_path}")

        from torch.cuda._memory_viz import trace_plot
        html_content = trace_plot(snapshot)
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"{tag}CUDA memory html visualization saved: {html_path}")

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self._recording = True
            logger.info("CUDA memory history recording started")

    def on_step_end(self, args, state, control, **kwargs):
        if self._recording and self.dump_interval and state.global_step % self.dump_interval == 0:
            self._dump_and_visualize(state.global_step, tag=f"step:{state.global_step}, ")

    def on_train_end(self, args, state, control, **kwargs):
        if self._recording:
            self._dump_and_visualize(state.global_step, tag="[final] ")
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording = False
