#!/bin/bash

# This script simplifies running READ-CLIP with Docker

# Default values
GPU_FLAG="all"
MODE="shell"
CONFIG_PATH=""
WANDB_KEY=""

# Parse command line arguments
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
      echo "Usage: ./script/run_docker.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --train                Run training mode"
      echo "  --eval                 Run evaluation mode"
      echo "  --config PATH          Path to configuration file"
      echo "  --wandb-key KEY        Weights & Biases API key"
      echo "  --gpu NUM              GPU device number (default: all)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create necessary directories
mkdir -p data
mkdir -p output
mkdir -p logs

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker could not be found. Please install Docker and try again."
    exit 1
fi

# Check if image exists, build if not
if [[ "$(docker images -q read-clip 2> /dev/null)" == "" ]]; then
    echo "Building READ-CLIP Docker image..."
    docker build -t read-clip .
fi

# Define command based on mode
if [[ "$MODE" == "train" ]]; then
    if [[ -z "$CONFIG_PATH" ]]; then
        CONFIG_PATH="config/train_read_clip.yaml"
    fi
    
    CMD="python train.py --cfg-path $CONFIG_PATH"
    
    if [[ ! -z "$WANDB_KEY" ]]; then
        CMD="$CMD --wandb-key $WANDB_KEY"
    fi
    
elif [[ "$MODE" == "eval" ]]; then
    if [[ -z "$CONFIG_PATH" ]]; then
        CONFIG_PATH="config/eval_read_clip.yaml"
    fi
    
    CMD="python evaluate.py --cfg-path $CONFIG_PATH"
    
else
    CMD="bash"
fi

# Run Docker container
echo "Running READ-CLIP in Docker..."
docker run --gpus \"device=$GPU_FLAG\" -it \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/logs:/app/logs" \
    read-clip $CMD

echo "Done!" 