from src.common.registry import registry
from src.tasks.base import (
    BaseTask,
    BaseTrainTask,
    BaseEvaluateTask,
    TaskWithPretrainedModel,
    TaskWithCustomModel
)
from src.tasks.train_task import (
    SingleTrainTask,
    DatasetSingleTrainTask,
    IterableDatasetSingleTrainTask,
    SingleTrainTaskWithCustomModel,
    SingleTrainTaskWithPretrainedModel,
    IterableDatasetTrainTaskWithCustomModel,
    IterableDatasetTrainTaskWithPretrainedModel,
    DatasetTrainTaskWithCustomModel,
    DatasetTrainTaskWithPretrainedModel,
)
from .evaluate_task import (
    MultiDatasetEvaluateTask,
    MultiDatasetEvaluateTaskWithPretrainedModel,
    MultiDatasetEvaluateTaskWithCustomModel,
)

__all__ = [
    "BaseTask",
    "BaseTrainTask",
    "BaseEvaluateTask",
    "TaskWithPretrainedModel",
    "TaskWithCustomModel",
    "SingleTrainTask",
    "DatasetSingleTrainTask",
    "IterableDatasetSingleTrainTask",
    "SingleTrainTaskWithCustomModel",
    "SingleTrainTaskWithPretrainedModel",
    "IterableDatasetTrainTaskWithCustomModel",
    "IterableDatasetTrainTaskWithPretrainedModel",
    "DatasetTrainTaskWithCustomModel",
    "DatasetTrainTaskWithPretrainedModel",
    "MultiDatasetEvaluateTask",
    "MultiDatasetEvaluateTaskWithPretrainedModel",
    "MultiDatasetEvaluateTaskWithCustomModel",
]


def setup_task(config):
    assert "task" in config.run_config, "Task name must be provided."

    task_name = config.run_config.task
    task = registry.get_task_class(task_name).setup_task(config=config)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task
