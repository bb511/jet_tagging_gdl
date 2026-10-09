# Logger that prefixes every message with the rank of the process that emitted it.

import logging

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A logger for multi-process runs, logging on rank 0 alone."""

    def __init__(self, name: str = __name__) -> None:
        super().__init__(logger=logging.getLogger(name), extra=None)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log *msg* at *level* on rank 0, prefixed with the rank.

        :raises RuntimeError: before Lightning has set ``rank_zero_only.rank``, without
            which the rank of this process is unknown.
        """
        if not self.isEnabledFor(level):
            return
        msg, kwargs = self.process(msg, kwargs)
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if current_rank == 0:
            self.logger.log(
                level, rank_prefixed_message(msg, current_rank), *args, **kwargs
            )
