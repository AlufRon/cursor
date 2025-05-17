#!/bin/bash

# --- Configuration ---
CONDA_ENV_NAME="moshi"
PYTHON_SCRIPT_PATH="/home/alufr/ttt_pytorch/ttt-lm-pytorch/train_ttt_pytorch.py" # Path to your PyTorch training script

# Model Configuration
MODEL_CONFIG_NAME="125m" # Or "350m", "760m", "1.3b", or path to a custom config JSON
TTT_LAYER_TYPE="linear"  # "linear" or "mlp"
NUM_CODEBOOKS=8
CODEBOOK_SIZE=2048
MAX_LENGTH=1024          # Sequence length for training chunks
TTT_BASE_LR=1.0
MINI_BATCH_SIZE_TTT=16
PRE_CONV=false           # Set to true to enable pre-convolution
CONV_KERNEL=4
USE_GATE=false           # Set to true to enable Mamba-like gating
SCAN_CHECKPOINT_GROUP_SIZE=0 # 0 for no scan checkpointing, TTT paper used 4 in JAX

# Data Paths
TOKENIZED_DATA_DIR="/sise/eliyanac-group/ron_al/librilight/unlab-6k_tokenize_full_length/"
VALIDATION_DATA_DIR="/sise/eliyanac-group/ron_al/librilight/unlab-6k_tokenize_full_length_test_clean/" # From previous context
CACHE_DIR_BASE="./mimi_data_cache_$(date +%Y%m%d_%H%M%S)" # Unique cache dir for each run
SKIP_INITIAL_TOKENS=0    # Number of initial *interleaved* tokens to skip from each audio file

# Training Hyperparameters
OUTPUT_DIR="./ttt_output_${MODEL_CONFIG_NAME}_${TTT_LAYER_TYPE}_$(date +%Y%m%d_%H%M%S)"
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=4 # Adjust based on your GPU memory
LEARNING_RATE=1e-3            # Peak learning rate
LR_SCHEDULER_TYPE="cosine"
WARMUP_STEPS_RATIO=0.1
WEIGHT_DECAY=0.1
GRADIENT_ACCUMULATION_STEPS=2 # Adjust to simulate larger batch sizes
MAX_GRAD_NORM=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPSILON=1e-8
MIXED_PRECISION="fp16"        # "no", "fp16", or "bf16"

# Logging & Checkpointing
LOGGING_STEPS=50
SAVE_STEPS=200               # Save checkpoints (and run validation) every X steps
SEED=42

# WandB (Optional)
USE_WANDB=true # Set to true to enable
WANDB_PROJECT="ttt-pytorch-librilight"
WANDB_RUN_NAME="${MODEL_CONFIG_NAME}-${TTT_LAYER_TYPE}-$(date +%Y%m%d_%H%M%S)"

# --- Activate Conda Environment ---
echo "Activating conda environment: $CONDA_ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh" # Ensure conda commands are available
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: $CONDA_ENV_NAME"
    exit 1
fi
echo "Conda environment activated."
echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"


# --- Construct Python Command ---
CMD="python $PYTHON_SCRIPT_PATH \
    --output_dir \"$OUTPUT_DIR\" \
    --model_config_name \"$MODEL_CONFIG_NAME\" \
    --ttt_layer_type \"$TTT_LAYER_TYPE\" \
    --num_codebooks $NUM_CODEBOOKS \
    --codebook_size $CODEBOOK_SIZE \
    --max_length $MAX_LENGTH \
    --ttt_base_lr $TTT_BASE_LR \
    --mini_batch_size_ttt $MINI_BATCH_SIZE_TTT \
    --conv_kernel $CONV_KERNEL \
    --scan_checkpoint_group_size $SCAN_CHECKPOINT_GROUP_SIZE \
    --tokenized_data_dir \"$TOKENIZED_DATA_DIR\" \
    --validation_data_dir \"$VALIDATION_DATA_DIR\" \
    --cache_dir_base \"$CACHE_DIR_BASE\" \
    --skip_initial_tokens $SKIP_INITIAL_TOKENS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type \"$LR_SCHEDULER_TYPE\" \
    --warmup_steps_ratio $WARMUP_STEPS_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --adam_epsilon $ADAM_EPSILON \
    --mixed_precision \"$MIXED_PRECISION\" \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --seed $SEED"

if [ "$PRE_CONV" = true ]; then
    CMD="$CMD --pre_conv"
fi

if [ "$USE_GATE" = true ]; then
    CMD="$CMD --use_gate"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\""
fi

# --- Execute Training ---
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" # Create output directory if it doesn't exist

echo "Running training command:"
echo "$CMD"

# Execute the command
eval "$CMD"

# --- Deactivate Conda Environment (Optional) ---
# echo "Deactivating conda environment."
# conda deactivate

echo "Training script finished."
