import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from src import (
    EvaluateConfig, setup_logger, setup_task, now, load_yml, get_rank, init_distributed_mode
)

# Initialize the logger
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument(
        "--cfg-path",
        required=True,  # Configuration path should be mandatory for correctness
        help="Path to the configuration file in YAML format."
    )

    return parser.parse_args()


def setup_seeds(seed: int) -> None:
    """
    Setup random seeds for reproducibility.

    Args:
        seed (int): Base seed value to initialize random number generators.
    """
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    logger.info(f"Random seed set to: {seed}")


def setup_file_logging(cfg_path: str, job_id: str) -> None:
    """
    Setup file logging for the evaluation job.

    Args:
        cfg_path (str): Path to the configuration file.
        job_id (str): Unique identifier for the job (e.g., timestamp).
    """
    file_name = os.path.basename(cfg_path).replace('.yml', '')
    log_dir = os.getenv("LOG_DIR", "./logs")  # Default to ./logs if LOG_DIR is not set
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{file_name}_{job_id}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file_path}")


def main() -> None:
    """
    Main function for evaluating a task based on a given configuration file.
    """
    try:
        # Generate a unique job ID for the current run
        job_id = now()

        # Setup console logging
        setup_logger()

        # Parse command-line arguments
        args = parse_args()

        # Setup file logging
        setup_file_logging(args.cfg_path, job_id)

        # Load evaluation configuration from YAML file
        evaluate_cfg = EvaluateConfig(**load_yml(args.cfg_path))

        # Initialize distributed mode if applicable
        init_distributed_mode(args)

        # Setup seeds for reproducibility
        setup_seeds(evaluate_cfg.run_config.seed)

        # Initialize the evaluation task
        task = setup_task(evaluate_cfg)

        # Build and run the evaluator
        evaluator = task.build_evaluator()
        evaluator.evaluate()

        logger.info("Evaluation completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        raise


if __name__ == "__main__":
    main()