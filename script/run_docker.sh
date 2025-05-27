#!/bin/bash

# This script automates building and running READ-CLIP Docker with train/eval/shell modes

set -e

# Default values
MODE="shell"
CONFIG_PATH=""
WANDB_KEY=""
GPU_FLAG="all"

# Argument parsing
while (( "$#" )); do
  case "$1" in
    --train)
      MODE="train"
      shift
      ;;
    --eval)
      MODE="eval"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --wandb-key)
      WANDB_KEY="$2"
      shift 2
      ;;
    --gpu)
      GPU_FLAG="$2"
      shift 2
      ;;
    --help)
      echo "Usage: bash run_docker.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --train                Run training mode"
      echo "  --eval                 Run evaluation mode"
      echo "  --config PATH          Path to configuration file"
      echo "  --wandb-key KEY        Weights & Biases API key"
      echo "  --gpu DEVICES          GPU device(s) to use (default: all, e.g. 0,1)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown parameter: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

REPO_DIR=$(pwd)
IMAGE_NAME="read-clip"
CONTAINER_NAME="read-clip-run"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Build Docker image if not present
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "[INFO] Building Docker image ($IMAGE_NAME)..."
    docker build -t $IMAGE_NAME .
fi

# Remove existing container if present
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "[INFO] Removing existing container ($CONTAINER_NAME)..."
    docker rm -f $CONTAINER_NAME
fi

# Prepare data/output/logs folders for volume mapping
mkdir -p data output logs

# Determine which command to run in container
if [[ "$MODE" == "train" ]]; then
    DOCKER_CMD="source /venv/bin/activate && chmod +x scripts/train.sh && bash scripts/train.sh"
    [ ! -z "$WANDB_KEY" ] && DOCKER_CMD="$DOCKER_CMD --wandb-key $WANDB_KEY"
elif [[ "$MODE" == "eval" ]]; then
    DOCKER_CMD="source /venv/bin/activate && chmod +x scripts/eval.sh && bash scripts/eval.sh"
else
    DOCKER_CMD="bash"
fi

echo "[INFO] Running READ-CLIP Docker ($MODE mode)..."
docker run --gpus "device=$GPU_FLAG" -it \
    --name $CONTAINER_NAME \
    -v "$REPO_DIR/data:/app/data" \
    -v "$REPO_DIR/output:/app/output" \
    -v "$REPO_DIR/logs:/app/logs" \
    $IMAGE_NAME \
    bash -c "$DOCKER_CMD"

echo "[INFO] Done!"