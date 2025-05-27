export HF_DATASETS_CACHE="../.cache"
export HF_HOME="../.cache"
export LOG_DIR="../.log"

DEVICES=0
NUM_TRAINERS=1

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
ROOT_DIR=$(dirname "$PARENT_DIR")

CFG_PATHS=(
    "$PARENT_DIR/config/train_read_clip.yaml"
)

for i in "${!CFG_PATHS[@]}"
do
    CFG_PATH="${CFG_PATHS[$i]}"

    echo "Currently Running with Config: $CFG_PATH"

    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --standalone \
        --nproc_per_node=$NUM_TRAINERS \
        "$PARENT_DIR/train.py" \
            --cfg-path "$CFG_PATH" \
            --wandb-key "$WANDB_KEY"
done