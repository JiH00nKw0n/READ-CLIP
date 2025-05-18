import logging
import src.utils.dist_utils as dist_utils
import os
import sys
from typing import Union, TextIO
from typing import Optional, Type
from types import TracebackType
from dataclasses import dataclass

__all__ = ["setup_logger", "Logger"]

LOG_DIR = os.getenv("LOG_DIR")
ROOT_LOGGER = 'MULTIMODAL'


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


@dataclass
class Logger:
    name: str = ROOT_LOGGER
    level: int = logging.DEBUG

    def __post_init__(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def addFileHandler(
            self,
            file_name: str,
            from_level: Optional[int] = logging.DEBUG,
            to_level: Optional[int] = logging.ERROR,
            formatter: str = "[%(asctime)s, %(levelname)s] : %(message)s"
    ):
        handler = logging.FileHandler(os.path.join(LOG_DIR, f"{file_name}.log"), encoding='utf-8')
        formatter = logging.Formatter(formatter)
        handler.setLevel(from_level)
        handler.setFormatter(formatter)

        level_filter = LevelFilter(from_level, to_level)
        handler.addFilter(level_filter)

        self.logger.addHandler(handler)

    def addStreamHandler(
            self,
            log_stream: Optional[TextIO] = sys.stdout,
            from_level: Optional[int] = logging.DEBUG,
            to_level: Optional[int] = logging.ERROR,
            formatter: str = "[%(asctime)s, %(levelname)s] : %(message)s"
    ):
        if log_stream is not None:
            handler = logging.StreamHandler(stream=log_stream)
        else:
            handler = logging.StreamHandler()
        formatter = logging.Formatter(formatter)
        handler.setLevel(from_level)
        handler.setFormatter(formatter)

        level_filter = LevelFilter(from_level, to_level)
        handler.addFilter(level_filter)

        self.logger.addHandler(handler)


class LevelFilter(logging.Filter):
    def __init__(self, low: Optional[int] = logging.DEBUG, high: Optional[int] = logging.ERROR):
        super().__init__()
        self.low = low if low is not None else logging.DEBUG
        self.high = high if high is not None else logging.ERROR

    def filter(self, record: logging.LogRecord) -> bool:
        return self.low <= record.levelno <= self.high


@dataclass
class LogContext:
    logger: Union[logging.Logger | Logger]
    local_rank: int
    level: int = logging.DEBUG
    default: int = logging.CRITICAL

    def __post_init__(self):
        self.original_level = self.logger.level

    def __enter__(self):
        if not self.local_rank == 0:
            self.logger.setLevel(self.default)
        else:
            self.logger.setLevel(self.level)

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType]
    ):
        self.logger.setLevel(self.original_level)
