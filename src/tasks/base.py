import logging
from typing import Any, Optional, Dict, Type, TypeVar

from datasets import Dataset, IterableDataset
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Extra
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin, add_end_docstrings

from src.common import EvaluateConfig, TrainConfig

# Type aliases for common types used in task classes
ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

# __all__ specifies which names are public and should be accessible via 'from module import *'
__all__ = [
    "BaseTask",
    "TaskWithPretrainedModel",
    "TaskWithCustomModel",
    "EVALUATE_TASK_DOCSTRING",
    "BaseEvaluateTask",
    "TRAIN_TASK_DOCSTRING",
    "BaseTrainTask",
]

# Setting up logger for debugging purposes
logger = logging.getLogger(__name__)

# Base Task docstring for reuse across classes
TASK_DOCSTRING = """
    A task class that defines the fundamental structure and workflow for different tasks involving models, processors, 
    and datasets. This class serves as the base class for task-specific implementations, providing an interface for 
    building models, processors, and datasets, as well as setting up tasks.

    The `BaseTask` class acts as a blueprint for other task-specific classes, ensuring a consistent interface across 
    different tasks. Each task class that inherits from `BaseTask` must override the `build_model`, `build_processor`, 
    and `build_datasets` methods.

    Args:
        config (`Any`):
            The configuration object for setting up the task, including model, processor, and dataset configurations.

    Methods:
        setup_task(config):
            A class method to initialize the task with the provided configuration.

        build_model(model_config: `Optional[Dict]`):
            Abstract method to build and return a model instance. Must be overridden by subclasses.

        build_processor(processor_config: `Optional[Dict]`):
            Abstract method to build and return a processor instance. Must be overridden by subclasses.

        build_datasets(dataset_config: `Optional[Dict]`, shuffle: `Optional[bool]`, buffer_size: `Optional[int]`):
            Abstract method to build and return datasets for the task. Must be overridden by subclasses.
"""


@add_end_docstrings(TASK_DOCSTRING)
class BaseTask(BaseModel):
    """
    The `BaseTask` is an abstract class that provides a template for task-specific classes. It defines methods for
    building models, processors, and datasets, which must be implemented by any class that inherits from `BaseTask`.

    This base class enforces consistency across tasks by requiring the implementation of core methods that set up
    the necessary components for task execution.
    """

    config: Any
    model_config = ConfigDict(extra=Extra.forbid)

    @classmethod
    def setup_task(cls, config: Any) -> "BaseTask":
        """
        Initialize and return an instance of the task using the provided configuration.

        Args:
            config (Any): The configuration for the task.

        Returns:
            BaseTask: An instance of the task.
        """
        return cls(config=config)

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        """
        Abstract method for building a model. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        """
        Abstract method for building a processor. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ) -> DatasetType:
        """
        Abstract method for building datasets. Must be implemented by subclasses.
        """
        raise NotImplementedError


@add_end_docstrings(TASK_DOCSTRING)
class TaskWithPretrainedModel(BaseTask):
    """
    A task class for handling pretrained models with optional LoRA configuration for parameter-efficient fine-tuning.
    This class extends the `BaseTask` and provides implementations for building a model and processor.

    Methods:
        build_model(model_config: `Optional[Dict]`):
            Builds and returns a pretrained model. If a LoRA configuration is provided, it applies LoRA to the model.

        build_processor(processor_config: `Optional[Dict]`):
            Builds and returns a processor for the task.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        """
        Builds a processor using the provided configuration. If no configuration is provided, it uses
        the processor configuration from `self.config`.

        Args:
            processor_config (`Optional[Dict]`, *optional*):
                The processor configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ProcessorType`: The processor instance.
        """
        processor_config = processor_config if processor_config is not None else self.config.processor_config.copy()
        return AutoProcessor.from_pretrained(**processor_config.config)


@add_end_docstrings(TASK_DOCSTRING)
class TaskWithCustomModel(BaseTask):
    """
    A task class for building a custom model and processor using configurations from a registry.
    This class supports models and processors that require custom configurations, and optionally
    supports LoRA-based parameter-efficient fine-tuning for text and vision models.

    Methods:
        build_model(model_config: `Optional[Dict]`):
            Builds and returns a custom model based on configurations from the registry.
            Supports LoRA fine-tuning for text and vision models.

        build_processor(processor_config: `Optional[Dict]`):
            Builds and returns a custom processor for text and vision tasks.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        raise NotImplementedError

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        """
        Builds a custom processor using the provided configuration from the registry.

        Args:
            processor_config (`Optional[Dict]`, *optional*):
                The processor configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ProcessorType`: The processor instance.

        Raises:
            AssertionError: If the processor class is not properly registered in the registry.
        """
        from src.common.registry import registry

        processor_config = processor_config if processor_config is not None else self.config.processor_config.copy()

        # Get the processor class from the registry
        processor_cls = registry.get_processor_class(processor_config.processor_cls)

        assert processor_cls is not None, "Processor {} not properly registered.".format(processor_cls)

        if hasattr(processor_cls, 'from_text_vision_pretrained'):
            return processor_cls.from_text_vision_pretrained(**processor_config.config)
        else:
            return processor_cls.from_pretrained(**processor_config.config)

# Evaluate Task docstring
EVALUATE_TASK_DOCSTRING = """
    A task class for evaluating models using predefined or custom evaluators and datasets. 
    The `EvaluateTask` class provides methods for building datasets and evaluators, and for adding 
    evaluators to the container. 

    Attributes:
        config (EvaluateConfig): The configuration for the evaluation task.

    Methods:
        build_datasets(dataset_config):
            Builds and returns datasets using the provided dataset configuration.

        build_evaluator(evaluator_config):
            Builds and returns evaluators using the provided evaluator configuration.
"""


@add_end_docstrings(EVALUATE_TASK_DOCSTRING)
class BaseEvaluateTask(BaseTask):
    """
    An abstract base class for tasks that evaluate models using evaluators and datasets.
    This class provides abstract methods for building datasets and evaluators, which must be
    implemented by subclasses. Classes that inherit from `EvaluateTask` will define specific
    evaluation logic by implementing the abstract methods.

    Inherits from:
        BaseTask: A base class that defines the common interface for all task types.
    """

    config: EvaluateConfig

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.

        Args:
            dataset_config (Optional[Dict], *optional*): The dataset configuration. Defaults to `None`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_evaluator(
            self,
            evaluator_config: Optional[DictConfig] = None
    ):
        """
        Abstract method for building evaluators. Must be implemented by subclasses.

        Args:
            evaluator_config (Optional[DictConfig], *optional*): The evaluator configuration. Defaults to `None`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


# Train Task docstring
TRAIN_TASK_DOCSTRING = """
    A task class for training models using predefined or custom datasets and trainers. 
    The `BaseTrainTask` class provides methods for building datasets and trainers, and for managing 
    the training process.

    Attributes:
        config (TrainConfig): The configuration for the training task.

    Methods:
        build_datasets(dataset_config, shuffle):
            Builds and returns datasets using the provided
    Methods:
        build_datasets(dataset_config, shuffle):
            Builds and returns datasets using the provided dataset configuration.

        build_trainer(trainer_config):
            Builds and returns a trainer using the provided trainer configuration.
"""


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class BaseTrainTask(BaseTask):
    """
    An abstract base class for tasks that train models using datasets and trainers.
    This class provides abstract methods for building datasets and trainers, which must be
    implemented by subclasses. Classes that inherit from `BaseTrainTask` will define specific
    training logic by implementing the abstract methods.

    Inherits from:
        BaseTask: A base class that defines the common interface for all task types.

    Attributes:
        config (TrainConfig): The configuration for the training task.
    """

    config: TrainConfig

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ):
        """
        Abstract method for building datasets. Must be implemented by subclasses.

        Args:
            dataset_config (Optional[Dict], *optional*): The dataset configuration. Defaults to `None`.
            shuffle (Optional[bool], *optional*): Whether to shuffle the dataset. Defaults to `False`.
            buffer_size (Optional[int], *optional*): Buffer size for shuffling. Defaults to `10000`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def build_trainer(
            self,
            trainer_config: Optional[DictConfig] = None
    ):
        """
        Abstract method for building the trainer. Must be implemented by subclasses.

        Args:
            trainer_config (Optional[DictConfig], *optional*): The trainer configuration. Defaults to `None`.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
