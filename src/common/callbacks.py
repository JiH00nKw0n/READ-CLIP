import logging

from transformers.integrations import WandbCallback

logger = logging.getLogger(__name__)

__all__ = ["CustomWandbCallback"]


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    individual_prefix = "ind_"
    individual_prefix_len = len(individual_prefix)

    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(individual_prefix):
            new_d["word/" + k[individual_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class CustomWandbCallback(WandbCallback):
    """
    A custom callback class that extends the functionality of `WandbCallback` from Hugging Face's Transformers library.
    This class integrates Weights and Biases (Wandb) logging into the training process, while allowing for custom behavior
    during the logging of metrics.

    Methods:
        on_log(args, state, control, model=None, logs=None, **kwargs):
            Custom behavior that occurs when logging metrics during training.
    """

    def __init__(self):
        """
        Initializes the `CustomWandbCallback` class by calling the parent `WandbCallback` class's constructor.
        Ensures Wandb integration is set up properly.
        """
        super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        Overrides the `on_log` method from `WandbCallback` to implement custom behavior when logging metrics
        during training. This method can be further extended to log additional information or perform
        actions based on the logged data.

        Args:
            args:
                The training arguments.
            state:
                The current training state, including information like the current step and number of completed epochs.
            control:
                Control flow object that can modify the behavior of training.
            model (Optional):
                The model being trained (if any).
            logs (Optional):
                A dictionary of metrics and logs that are being logged at the current step.
            **kwargs:
                Additional keyword arguments passed to the function.

        This method calls the parent class's `on_log` method to ensure Wandb logging functionality remains intact.
        """
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
