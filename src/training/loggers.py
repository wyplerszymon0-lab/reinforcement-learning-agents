"""
Optional scalar logging for the trainers.

The trainers call `logger.scalar(tag, value, step)` with the environment step as
`step`, so DQN and PPO curves share an x-axis. Without a logger nothing is written
beyond the usual JSON results. TensorBoardLogger needs the `tensorboard` package
and imports it only when it is created.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ScalarLogger(Protocol):
    def scalar(self, tag: str, value: float, step: int) -> None: ...

    def close(self) -> None: ...


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("--tensorboard needs the tensorboard package: pip install tensorboard") from exc
        self.log_dir = Path(log_dir)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, global_step=step)

    def close(self) -> None:
        self._writer.close()
