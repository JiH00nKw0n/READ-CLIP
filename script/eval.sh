export HF_DATASETS_CACHE="../.cache"
export HF_HOME="../.cache"
export LOG_DIR="../.log"
export TOKENIZERS_PARALLELISM=false

DEVICES=0
NUM_TRAINERS=1

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ROOT_DIR=$(dirname "$PARENT_DIR")

CFG_PATHS=(
    "$PARENT_DIR/config/eval_read_clip.yaml"
)

for CFG_PATH in "${CFG_PATHS[@]}"; do
    echo "Running with config: $CFG_PATH"
    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --standalone \
        --nproc_per_node=$NUM_TRAINERS \
        "$PARENT_DIR/evaluate.py" \
        --cfg-path "$CFG_PATH"
done
