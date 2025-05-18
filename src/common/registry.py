from typing import Type, TypeVar
# from src.collators.base import BaseCollator
# from src.runners import BaseTrainer, BaseEvaluator
# from src.tasks import BaseTask
from transformers import ProcessorMixin, PretrainedConfig, PreTrainedModel, Trainer

ProcessorType = Type[ProcessorMixin]
CollatorType = TypeVar("CollatorType", bound="BaseCollator")
TaskType = TypeVar("TaskType", bound="BaseTask")
ModelType = Type[PreTrainedModel]
ModelConfigType = Type[PretrainedConfig]
TrainerType = TypeVar("TrainerType", bound="BaseTrainer")
BuilderType = TypeVar("BuilderType", bound="BaseBuilder")
EvaluatorType = TypeVar("EvaluatorType", bound="BaseEvaluator")


class Registry:
    """
    A registry class that allows for registering and retrieving different types of components
    such as builders, tasks, processors, collators, models, trainers, and evaluators. It also provides
    methods for listing and retrieving registered classes.

    The registry stores mappings of component names to their respective classes, ensuring that names
    are unique and providing utilities to access registered classes by name.

    Attributes:
        mapping (`dict`): A dictionary that holds mappings for various component types.
    """

    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "collator_name_mapping": {},
        "model_name_mapping": {},
        "model_config_name_mapping": {},
        "trainer_name_mapping": {},
        "evaluator_name_mapping": {},
    }

    @classmethod
    def register_processor(cls, name):
        """
        Registers a processor class by its name.

        Args:
            name (`str`): The name to register the processor class with.

        Returns:
            A decorator that registers the processor class.
        """

        def wrap(processor_cls) -> ProcessorType:

            assert issubclass(
                processor_cls, ProcessorMixin
            ), "All processors must inherit ProcessorMixin"

            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['processor_name_mapping'][name]}."
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls

            return processor_cls

        return wrap

    @classmethod
    def register_collator(cls, name):
        """
        Registers a collator class by its name.

        Args:
            name (`str`): The name to register the collator class with.

        Returns:
            A decorator that registers the collator class.
        """

        def wrap(collator_cls) -> CollatorType:
            from src.collators import BaseCollator
            assert issubclass(
                collator_cls, BaseCollator
            ), "All collators must inherit BaseCollator"

            if name in cls.mapping["collator_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['collator_name_mapping'][name]}."
                )
            cls.mapping["collator_name_mapping"][name] = collator_cls

            return collator_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        """
        Registers a task class by its name.

        Args:
            name (`str`): The name to register the task class with.

        Returns:
            A decorator that registers the task class.
        """

        def wrap(task_cls) -> TaskType:
            from src.tasks import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask"

            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['task_name_mapping'][name]}."
                )
            cls.mapping["task_name_mapping"][name] = task_cls

            return task_cls

        return wrap

    @classmethod
    def register_trainer(cls, name):
        """
        Registers a trainer class by its name.

        Args:
            name (`str`): The name to register the trainer class with.

        Returns:
            A decorator that registers the trainer class.
        """

        def wrap(trainer_cls) -> TrainerType:

            assert issubclass(
                trainer_cls, Trainer
            ), "All trainers must inherit Trainer"

            if name in cls.mapping["trainer_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['trainer_name_mapping'][name]}."
                )
            cls.mapping["trainer_name_mapping"][name] = trainer_cls

            return trainer_cls

        return wrap

    @classmethod
    def register_evaluator(cls, name):
        """
        Registers an evaluator class by its name.

        Args:
            name (`str`): The name to register the evaluator class with.

        Returns:
            A decorator that registers the evaluator class.
        """

        def wrap(evaluator_cls) -> EvaluatorType:
            from src.runners.evaluator import BaseEvaluator

            assert issubclass(
                evaluator_cls, BaseEvaluator
            ), "All evaluators must inherit BaseEvaluator"

            if name in cls.mapping["evaluator_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['evaluator_name_mapping'][name]}."
                )
            cls.mapping["evaluator_name_mapping"][name] = evaluator_cls

            return evaluator_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        """
        Registers a model class by its name.

        Args:
            name (`str`): The name to register the model class with.

        Returns:
            A decorator that registers the model class.
        """

        def wrap(model_cls) -> ModelType:

            assert issubclass(
                model_cls, PreTrainedModel
            ), "All models must inherit PreTrainedModel"

            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['model_name_mapping'][name]}."
                )
            cls.mapping["model_name_mapping"][name] = model_cls

            return model_cls

        return wrap

    @classmethod
    def register_model_config(cls, name):
        """
        Registers a model configuration class by its name.

        Args:
            name (`str`): The name to register the model configuration class with.

        Returns:
            A decorator that registers the model configuration class.
        """

        def wrap(config_cls) -> ModelConfigType:

            assert issubclass(
                config_cls, PretrainedConfig
            ), "All model configurations must inherit PretrainedConfig"

            if name in cls.mapping["model_config_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['model_config_name_mapping'][name]}."
                )
            cls.mapping["model_config_name_mapping"][name] = config_cls

            return config_cls

        return wrap

    @classmethod
    def register_builder(cls, name):
        """
        Registers a builder class by its name.

        Args:
            name (`str`): The name to register the builder class with.

        Returns:
            A decorator that registers the builder class.
        """

        def wrap(builder_cls) -> BuilderType:
            from src.datasets.base import BaseBuilder

            assert issubclass(
                builder_cls, BaseBuilder
            ), "All builders must inherit BaseBuilder"

            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    f"Name '{name}' already registered for {cls.mapping['builder_name_mapping'][name]}."
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls

            return builder_cls

        return wrap

    @classmethod
    def get_builder_class(cls, name):
        """
        Retrieves the builder class registered under the given name.

        Args:
            name (`str`): The name of the builder class to retrieve.

        Returns:
            The builder class if registered, otherwise `None`.
        """
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        """
        Retrieves the model class registered under the given name.

        Args:
            name (`str`): The name of the model class to retrieve.

        Returns:
            The model class if registered, otherwise `None`.
        """
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_model_config_class(cls, name):
        """
        Retrieves the model configuration class registered under the given name.

        Args:
            name (`str`): The name of the model configuration class to retrieve.

        Returns:
            The model configuration class if registered, otherwise `None`.
        """
        return cls.mapping["model_config_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        """
        Retrieves the task class registered under the given name.

        Args:
            name (`str`): The name of the task class to retrieve.

        Returns:
            The task class if registered, otherwise `None`.
        """
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        """
        Retrieves the processor class registered under the given name.

        Args:
            name (`str`): The name of the processor class to retrieve.

        Returns:
            The processor class if registered, otherwise `None`.
        """
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_collator_class(cls, name):
        """
        Retrieves the collator class registered under the given name.

        Args:
            name (`str`): The name of the collator class to retrieve.

        Returns:
            The collator class if registered, otherwise `None`.
        """
        return cls.mapping["collator_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name):
        """
        Retrieves the trainer class registered under the given name.

        Args:
            name (`str`): The name of the trainer class to retrieve.

        Returns:
            The trainer class if registered, otherwise `None`.
        """
        return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_evaluator_class(cls, name):
        """
        Retrieves the evaluator class registered under the given name.

        Args:
            name (`str`): The name of the evaluator class to retrieve.

        Returns:
            The evaluator class if registered, otherwise `None`.
        """
        return cls.mapping["evaluator_name_mapping"].get(name, None)

    @classmethod
    def list_trainers(cls):
        """
        Lists all registered trainer class names.

        Returns:
            A sorted list of trainer class names.
        """
        return sorted(cls.mapping["trainer_name_mapping"].keys())

    @classmethod
    def list_evaluators(cls):
        """
        Lists all registered evaluator class names.

        Returns:
            A sorted list of evaluator class names.
        """
        return sorted(cls.mapping["evaluator_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        """
        Lists all registered model class names.

        Returns:
            A sorted list of model class names.
        """
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_model_configs(cls):
        """
        Lists all registered model configuration class names.

        Returns:
            A sorted list of model configuration class names.
        """
        return sorted(cls.mapping["model_config_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        """
        Lists all registered task class names.

        Returns:
            A sorted list of task class names.
        """
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        """
        Lists all registered processor class names.

        Returns:
            A sorted list of processor class names.
        """
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_collators(cls):
        """
        Lists all registered collator class names.

        Returns:
            A sorted list of collator class names.
        """
        return sorted(cls.mapping["collator_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        """
        Lists all registered dataset builder class names.

        Returns:
            A sorted list of dataset builder class names.
        """
        return sorted(cls.mapping["builder_name_mapping"].keys())


registry = Registry()
