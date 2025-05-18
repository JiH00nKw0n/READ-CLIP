import logging
from typing import Optional, Dict, Type, List

from datasets import Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from transformers import add_end_docstrings, PreTrainedModel, ProcessorMixin

from src.common import registry, experimental
from src.datasets import BaseBuilder
from src.runners import BaseEvaluator
from src.tasks.base import (
    TaskWithPretrainedModel, TaskWithCustomModel, EVALUATE_TASK_DOCSTRING, BaseEvaluateTask
)
from src.utils import load_json

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
        evaluate(batch_size):
            Runs the evaluation for all the evaluators in the container.

        add(evaluator):
            Adds a new evaluator to the container.
    """

    container: Optional[List[EvaluatorType]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False, strict=False, validate_assignment=False)

    def evaluate(self, batch_size: Optional[int] = 128):
        """
        Runs the evaluation process for each evaluator in the container.

        Args:
            batch_size (int, optional): The batch size for evaluation. Defaults to 128.
        """
        for evaluator in self.container:
            if issubclass(type(evaluator), BaseEvaluator):
                evaluator.evaluate(batch_size=batch_size)

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

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None
    ) -> Dict[str, BuilderType]:
        """
        Builds the datasets based on the provided configuration. The datasets are stored as builders keyed by their names.

        Args:
            dataset_config (Optional[Dict]): The dataset configuration.
            If not provided, defaults to `self.config.dataset_config`.

        Returns:
            Dict[str, BuilderType]: A dictionary containing dataset builders keyed by dataset name.
        """
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        builder_dict = {}
        assert len(dataset_config) > 0, "At least one dataset must be specified."

        for dataset_line in dataset_config:
            builder_cls_name, config = next(iter(dataset_line.items()))
            builder = registry.get_builder_class(builder_cls_name)(**config)
            builder_dict[builder_cls_name] = builder

        return builder_dict

    def build_evaluator(
            self,
            evaluator_config: Optional[DictConfig] = None
    ) -> EvaluatorContainer:
        """
        Builds and initializes the evaluators using the provided configuration. The evaluators are added to an
        `EvaluatorContainer` and evaluated on the corresponding datasets.

        Args:
            evaluator_config (Optional[DictConfig]): The evaluator configuration.
            If not provided, defaults to `self.config.evaluator_config`.

        Returns:
            EvaluatorContainer: A container holding multiple evaluators that can run evaluations for the model.
        """
        evaluator_config = evaluator_config if evaluator_config is not None else self.config.evaluator_config
        dataset_dict = self.build_datasets()

        container = EvaluatorContainer()
        model = self.build_model()
        processor = self.build_processor()

        for evaluator_line, (builder_cls_name, builder) in zip(
                evaluator_config,
                dataset_dict.items()
        ):

            evaluator_cls_name, config = next(iter(evaluator_line.items()))
            evaluator_cls = registry.get_evaluator_class(evaluator_cls_name)
            assert evaluator_cls is not None, f"Evaluator {evaluator_cls_name} not properly registered."

            collator_config = OmegaConf.create(config.pop('collator'))
            collator_cls = registry.get_collator_class(collator_config.collator_cls)
            assert collator_cls is not None, f"Collator {collator_cls} not properly registered."

            collator = collator_cls(
                processor=processor,
                **collator_config.config,
            )

            evaluate_dataset = builder.build_dataset()

            if not (isinstance(evaluate_dataset, Dataset) or isinstance(evaluate_dataset, DatasetDict)):
                raise TypeError(f"Expected `evaluate_dataset` to be of type `datasets.Dataset`, "
                                f"or `datasets.DatasetDict` but got {type(evaluate_dataset)} instead.")

            container.add(
                evaluator=evaluator_cls(
                    model=model,
                    evaluate_dataset=evaluate_dataset,
                    dataset_name=builder.name,
                    data_collator=collator,
                    **config
                )
            )

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

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None
    ) -> Dict[str, BuilderType]:
        return MultiDatasetEvaluateTask.build_datasets(
            self,
            dataset_config=dataset_config,
        )

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

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None
    ) -> Dict[str, BuilderType]:
        return MultiDatasetEvaluateTask.build_datasets(
            self,
            dataset_config=dataset_config,
        )

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
