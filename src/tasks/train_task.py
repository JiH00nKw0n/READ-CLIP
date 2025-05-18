import logging
import os
from typing import Optional, Dict, Type, TypeVar

from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    PreTrainedModel,
    ProcessorMixin,
    TrainingArguments,
    add_end_docstrings
)
from peft import get_peft_model, LoraConfig

from src.common import registry, TrainConfig
from src.tasks.base import BaseTrainTask, TaskWithCustomModel, TaskWithPretrainedModel, TRAIN_TASK_DOCSTRING
from src.utils import load_yml

ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
    "SingleTrainTask",
    "SingleTrainTaskWithPretrainedModel",
    "SingleTrainTaskWithCustomModel",
    "DatasetSingleTrainTask",
    "IterableDatasetSingleTrainTask",
    "DatasetTrainTaskWithPretrainedModel",
    "IterableDatasetTrainTaskWithPretrainedModel",
    "DatasetTrainTaskWithCustomModel",
    "IterableDatasetTrainTaskWithCustomModel",
]

logger = logging.getLogger(__name__)


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTask(BaseTrainTask):
    config: TrainConfig

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config

        collator_cls = registry.get_collator_class(self.config.collator_config.collator_cls)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_cls)

        train_dataset = self.build_datasets()

        collator = collator_cls(
            processor=self.build_processor(),
            **self.config.collator_config.config,
        )

        return trainer_cls(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class DatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, Dataset):
                raise TypeError("DatasetTrainTask must build dataset with `Dataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return concatenate_datasets(datasets)


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class IterableDatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, IterableDataset):
                raise TypeError("DatasetTrainTask must build dataset with `IterableDataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return interleave_datasets(datasets).with_format("torch")


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithPretrainedModel(SingleTrainTask, TaskWithPretrainedModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds a pretrained model using the provided configuration. If a LoRA configuration is specified,
        it applies the LoRA configuration to the model for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The model instance loaded with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is neither a string nor a valid path.
        """
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        # Initialize the model

        model = model_cls.from_pretrained(**model_config.config)

        if model_config.lora is not None:
            if isinstance(model_config.lora, str):
                lora_config = load_yml(model_config.lora)
            elif isinstance(model_config.lora, os.PathLike):
                lora_config = load_yml(os.fspath(model_config.lora))
            else:
                raise TypeError("`lora` configuration must be either a string or a valid path.")

            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)

            logger.info(f"{repr(model)}")
            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info(
                f'ALL PARAM: {all_param} / TRAINABLE PARAM: {trainable_params} / '
                f'RATIO: {trainable_params / all_param * 100}%'
            )

        return model.cuda().train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithCustomModel(SingleTrainTask, TaskWithCustomModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """
        Builds a custom model using the provided configuration from the registry. If a LoRA configuration
        is provided for either text or vision models, it applies the configuration for parameter-efficient fine-tuning.

        Args:
            model_config (`Optional[Dict]`, *optional*):
                The model configuration dictionary. If not provided, uses the configuration from `self.config`.

        Returns:
            `ModelType`: The custom model instance with optional LoRA fine-tuning.

        Raises:
            TypeError: If `model_config.lora` is not a valid `DictConfig` object.
        """

        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        # Initialize the model configuration and model
        model_cfg = model_cfg_cls(**model_config.config)
        model = model_cls(model_cfg)

        # Apply LoRA configurations if provided
        if model_config.lora is not None:
            if isinstance(model_config.lora, DictConfig):
                text_model_config_path = model_config.lora.pop('text_model', None)
                vision_model_config_path = model_config.lora.pop('vision_model', None)
            else:
                raise TypeError("LoRA configuration must be a valid `DictConfig` object.")

            # Apply LoRA configuration to text model
            if text_model_config_path is not None:
                text_model_lora_config = load_yml(text_model_config_path)
                text_model_peft_config = LoraConfig(**text_model_lora_config)
                model.text_model = get_peft_model(model.text_model, text_model_peft_config)

            # Apply LoRA configuration to vision model
            if vision_model_config_path is not None:
                vision_model_lora_config = load_yml(vision_model_config_path)
                vision_model_peft_config = LoraConfig(**vision_model_lora_config)
                model.vision_model = get_peft_model(model.vision_model, vision_model_peft_config)

        return model.cuda().train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithPretrainedModel")
class DatasetTrainTaskWithPretrainedModel(DatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithPretrainedModel")
class IterableDatasetTrainTaskWithPretrainedModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithCustomModel")
class DatasetTrainTaskWithCustomModel(DatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithCustomModel")
class IterableDatasetTrainTaskWithCustomModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )
