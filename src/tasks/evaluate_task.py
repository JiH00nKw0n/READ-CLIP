import logging
from typing import Optional, Dict, Type, List

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from transformers import add_end_docstrings, PreTrainedModel, ProcessorMixin

from src.builders import BaseBuilder
from src.common import registry, experimental
from src.runners import BaseEvaluator
from src.tasks.base import (
    TaskWithPretrainedModel, TaskWithCustomModel, EVALUATE_TASK_DOCSTRING, BaseEvaluateTask
)

__all__ = [
    "MultiDatasetEvaluateTask",
    "MultiDatasetEvaluateTaskWithPretrainedModel",
    "MultiDatasetEvaluateTaskWithCustomModel"
]

# Type aliases for common types
ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
EvaluatorType = Type[BaseEvaluator]
BuilderType = Type[BaseBuilder]

# Setting up logger for debugging purposes
logger = logging.getLogger(__name__)


@experimental
class EvaluatorContainer(BaseModel):
    """
    A container for holding multiple evaluators. This class allows for adding evaluators
    and evaluating them in a batch.

    Attributes:
        container (Optional[List[EvaluatorType]]): A list of evaluators to be added to the container.

    Methods:
        evaluate():
            Runs the evaluation for all the evaluators in the container.

        add(evaluator):
            Adds a new evaluator to the container.
    """

    container: Optional[List[EvaluatorType]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False)

    def evaluate(self):
        """
        Runs the evaluation process for each evaluator in the container.

        Args:
            batch_size (int, optional): The batch size for evaluation. Defaults to 128.
        """
        for evaluator in self.container:
            if issubclass(type(evaluator), BaseEvaluator):
                evaluator.evaluate()

    def add(self, evaluator: EvaluatorType):
        """
        Adds an evaluator to the container.

        Args:
            evaluator (EvaluatorType): The evaluator to add to the container.
        """
        self.container.append(evaluator)


@add_end_docstrings(EVALUATE_TASK_DOCSTRING)
class MultiDatasetEvaluateTask(BaseEvaluateTask):
    """
    A class for managing and running multiple evaluation tasks on a single model using different evaluators and datasets.
    This class allows for evaluating a single model across multiple evaluation tasks, each managed by an
    `EvaluatorContainer`.

    It provides methods for building datasets and evaluators, and executes multiple evaluation tasks on the model.
    """

    def build_evaluator(self, evaluator_config: Optional[DictConfig] = None) -> EvaluatorContainer:
        evaluator_config = evaluator_config or self.config.evaluator_config

        container = EvaluatorContainer()
        model = self.build_model()
        processor = self.build_processor()

        base_config = self.config.run_config.base_config

        collator_config = OmegaConf.create(base_config.pop('collator'))
        collator_cls = registry.get_collator_class(collator_config.collator_cls)
        assert collator_cls is not None, f"Collator {collator_cls} not properly registered."

        collator = collator_cls(
            processor=processor,
            **collator_config.config,
        )

        for evaluator_name in evaluator_config:
            evaluator_cls = registry.get_evaluator_class(evaluator_name)

            if not evaluator_cls:
                logger.error(f"Evaluator {evaluator_name} not registered.")
                raise ValueError(f"Evaluator {evaluator_name} not registered.")
            container.add(
                evaluator=evaluator_cls(
                    model=model,
                    data_collator=collator,
                    **base_config
                )
            )
        logger.info("Evaluator container successfully built.")
        return container


@add_end_docstrings(EVALUATE_TASK_DOCSTRING)
@registry.register_task("MultiDatasetEvaluateTaskWithPretrainedModel")
class MultiDatasetEvaluateTaskWithPretrainedModel(MultiDatasetEvaluateTask, TaskWithPretrainedModel):
    """
    An evaluation task for pretrained models. Inherits from `EvaluateTask` and `TaskWithPretrainedModel`.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> PreTrainedModel:
        """
        Builds and returns a pretrained model for evaluation. Optionally applies LoRA configurations.

        Args:
            model_config (Optional[Dict]): The model configuration.
            If not provided, defaults to `self.config.model_config`.

        Returns:
            PreTrainedModel: The pretrained model for evaluation.
        """
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, f"Model {model_cls} not properly registered."

        model = model_cls.from_pretrained(**model_config.config)

        return model.cuda().eval()

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return TaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_evaluator(
            self,
            evaluator_config: Optional[DictConfig] = None
    ) -> EvaluatorContainer:
        return MultiDatasetEvaluateTask.build_evaluator(
            self,
            evaluator_config=evaluator_config,
        )


@add_end_docstrings(EVALUATE_TASK_DOCSTRING)
@registry.register_task("MultiDatasetEvaluateTaskWithCustomModel")
class MultiDatasetEvaluateTaskWithCustomModel(MultiDatasetEvaluateTask, TaskWithCustomModel):
    """
    An evaluation task for custom models. Inherits from `EvaluateTask` and `TaskWithCustomModel`.
    """

    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        """
        Builds and returns a custom model for evaluation.

        Args:
            model_config (Optional[Dict]): The model configuration.
            If not provided, defaults to `self.config.model_config`.

        Returns:
            PreTrainedModel: The custom model for evaluation.
        """
        model_config = model_config \
            if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, f"Model {model_cls} not properly registered."

        model = model_cls.from_pretrained(**model_config.config)

        return model.cuda().eval()

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return TaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_evaluator(
            self,
            evaluator_config: Optional[DictConfig] = None
    ) -> EvaluatorContainer:
        return MultiDatasetEvaluateTask.build_evaluator(
            self,
            evaluator_config=evaluator_config,
        )
