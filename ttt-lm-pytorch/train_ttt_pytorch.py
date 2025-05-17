# train_ttt_pytorch.py
import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Assuming your TTT model definitions are in ttt_model.py
from ttt import TTTConfig, TTTForCausalLM 

# Import the new dataset
from mimi_audio_dataset import MimiInterleavedDataset # Make sure mimi_audio_dataset.py is in the same dir or PYTHONPATH

# Placeholder for WandB - uncomment and configure if you use it
# import wandb

# --- Configuration & Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def set_seed(seed_val=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    Basic collate function. Assumes all items in the batch have the same sequence length.
    No padding is handled here as MimiInterleavedDataset produces fixed-length chunks.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Test-Time Training (TTT) model with PyTorch for Mimi Audio Tokens.")
    
    # Model Configuration
    parser.add_argument("--model_config_name", type=str, default="125m", help="Standard model config name (e.g., 125m, 350m) or path to a custom config JSON.")
    parser.add_argument("--ttt_layer_type", type=str, default="linear", choices=["linear", "mlp"], help="TTT layer type.")
    # vocab_size will be determined by num_codebooks * codebook_size
    parser.add_argument("--num_codebooks", type=int, default=8, help="Number of codebooks for Mimi tokens.")
    parser.add_argument("--codebook_size", type=int, default=2048, help="Size of each codebook for Mimi tokens.")
    parser.add_argument("--hidden_size", type=int, help="Hidden size (overrides standard config if set).")
    parser.add_argument("--intermediate_size", type=int, help="Intermediate MLP size (overrides standard config if set).")
    parser.add_argument("--num_hidden_layers", type=int, help="Number of hidden layers (overrides standard config if set).")
    parser.add_argument("--num_attention_heads", type=int, help="Number of attention heads (overrides standard config if set).")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for training chunks.") # Renamed from max_position_embeddings
    parser.add_argument("--ttt_base_lr", type=float, default=1.0, help="Base learning rate for the TTT inner loop.")
    parser.add_argument("--mini_batch_size_ttt", type=int, default=16, help="Mini-batch size for TTT internal updates.")
    parser.add_argument("--pre_conv", action="store_true", help="Whether to use convolution before TTT layer.")
    parser.add_argument("--conv_kernel", type=int, default=4, help="Kernel size for pre_conv.")
    parser.add_argument("--use_gate", action="store_true", help="Whether to use gating in Mamba-like backbone.")
    parser.add_argument("--scan_checkpoint_group_size", type=int, default=0, help="Gradient checkpoint group size for TTT scan, 0 for no checkpointing.")


    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU for training.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Peak learning rate for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant"], help="Learning rate scheduler type.")
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1, help="Ratio of total training steps for linear warmup.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision training.")
    
    # Data & Tokenizer
    parser.add_argument("--tokenized_data_dir", type=str, required=True, help="Path to the directory containing pre-tokenized .pt files for training.")
    parser.add_argument("--validation_data_dir", type=str, default=None, help="Path to a separate directory for validation .pt files (overrides tokenized_data_dir/val).")
    parser.add_argument("--cache_dir_base", type=str, default="./data_cache", help="Base directory to cache processed token data.")
    parser.add_argument("--skip_initial_tokens", type=int, default=0, help="Number of initial *interleaved* tokens to skip from each audio file during processing.")


    # Logging & Checkpointing
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training information every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # WandB (optional)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--wandb_project", type=str, default="ttt-pytorch-mimi-training", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")

    return parser.parse_args()

# --- Main Training Function ---
def train():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir_base:
        Path(args.cache_dir_base).mkdir(parents=True, exist_ok=True)


    # Initialize WandB (optional)
    if args.use_wandb:
        # import wandb # Make sure wandb is installed
        # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
        logger.info("WandB logging enabled. Ensure you have logged in.")
    else:
        logger.info("WandB logging disabled.")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Effective vocabulary size for Mimi tokens
    effective_vocab_size = args.num_codebooks * args.codebook_size
    logger.info(f"Effective vocabulary size: {effective_vocab_size} ({args.num_codebooks} codebooks * {args.codebook_size} size)")


    # Model Configuration
    TTT_STANDARD_CONFIGS_PYTORCH = {
        "125m": {"hidden_size": 768, "intermediate_size": 2048, "num_hidden_layers": 12, "num_attention_heads": 12},
        "350m": {"hidden_size": 1024, "intermediate_size": 2736, "num_hidden_layers": 24, "num_attention_heads": 16},
        "760m": {"hidden_size": 1536, "intermediate_size": 4096, "num_hidden_layers": 24, "num_attention_heads": 16},
        "1.3b": {"hidden_size": 2048, "intermediate_size": 5504, "num_hidden_layers": 24, "num_attention_heads": 32},
    }
    
    if args.model_config_name in TTT_STANDARD_CONFIGS_PYTORCH:
        config_params = TTT_STANDARD_CONFIGS_PYTORCH[args.model_config_name]
        model_config = TTTConfig(
            vocab_size=effective_vocab_size, # Use effective vocab size
            hidden_size=args.hidden_size if args.hidden_size else config_params["hidden_size"],
            intermediate_size=args.intermediate_size if args.intermediate_size else config_params["intermediate_size"],
            num_hidden_layers=args.num_hidden_layers if args.num_hidden_layers else config_params["num_hidden_layers"],
            num_attention_heads=args.num_attention_heads if args.num_attention_heads else config_params["num_attention_heads"],
            max_position_embeddings=args.max_length, # TTT model expects max_position_embeddings
            ttt_layer_type=args.ttt_layer_type,
            ttt_base_lr=args.ttt_base_lr,
            mini_batch_size=args.mini_batch_size_ttt,
            pre_conv=args.pre_conv,
            conv_kernel=args.conv_kernel,
            use_gate=args.use_gate,
            # pad_token_id, bos_token_id, eos_token_id might not be relevant for raw token IDs
            # but TTTConfig expects them. Set to placeholder if not used by model logic.
            pad_token_id=0, # Placeholder
            bos_token_id=1, # Placeholder
            eos_token_id=2, # Placeholder
            scan_checkpoint_group_size=args.scan_checkpoint_group_size,
        )
    elif os.path.exists(args.model_config_name): # Path to a config.json
        model_config = TTTConfig.from_pretrained(args.model_config_name)
        model_config.vocab_size = effective_vocab_size
        model_config.max_position_embeddings = args.max_length
        model_config.ttt_layer_type = args.ttt_layer_type
        # ... (add other overrides as necessary)
    else:
        raise ValueError(f"Model config '{args.model_config_name}' not found or not a standard config name.")

    logger.info(f"Initializing model with config: {model_config}")
    model = TTTForCausalLM(model_config)
    model.to(device)

    # Dataset and DataLoader
    logger.info("Preparing datasets...")
    train_dataset = MimiInterleavedDataset(
        data_dir=args.tokenized_data_dir,
        split="train",
        max_length=args.max_length,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        cache_dir_base=args.cache_dir_base,
        skip_initial_tokens=args.skip_initial_tokens
    )
    # It's good practice to have a validation set
    val_dataset = MimiInterleavedDataset(
        data_dir=args.tokenized_data_dir, # Base dir, override will be used if provided
        split="validation",
        max_length=args.max_length,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        cache_dir_base=args.cache_dir_base,
        skip_initial_tokens=args.skip_initial_tokens,
        validation_override_dir=args.validation_data_dir
    )

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Please check your data paths and processing logic.")
        return
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty. Evaluation will be skipped or may fail.")


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4 # Adjust as needed
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_train_batch_size, # Can be different for eval
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 # Adjust as needed
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay
    )

    # Learning Rate Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_steps_ratio)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(max_train_steps - current_step) / float(max(1, max_train_steps - num_warmup_steps))
        )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=(args.mixed_precision == "fp16"))

    # Training Loop
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.time()
        model.train()
        train_epoch_loss = 0.0
        num_train_batches = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=(args.mixed_precision in ["fp16", "bf16"]), dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False 
                )
                loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            train_epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_train_batches +=1


            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_train_loss_recent = train_epoch_loss / (num_train_batches * args.gradient_accumulation_steps) # Avg loss since last log or start of epoch
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch: {epoch+1}/{args.num_train_epochs}, Step: {global_step}/{max_train_steps}, LR: {current_lr:.2e}, Train Loss (recent): {avg_train_loss_recent:.4f}")
                    if args.use_wandb:
                        # wandb.log({"train_loss_step": avg_train_loss_recent, "learning_rate": current_lr, "global_step": global_step})
                        pass 
                
                if global_step % args.save_steps == 0:
                    # --- Evaluation Step ---
                    model.eval()
                    val_epoch_loss = 0.0
                    num_val_batches = 0
                    if len(val_dataloader) > 0: # Only eval if val_dataloader is not empty
                        with torch.no_grad():
                            for val_batch in val_dataloader:
                                val_input_ids = val_batch["input_ids"].to(device)
                                val_attention_mask = val_batch["attention_mask"].to(device)
                                val_labels = val_batch["labels"].to(device)
                                
                                with autocast(enabled=(args.mixed_precision in ["fp16", "bf16"]), dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16):
                                    val_outputs = model(
                                        input_ids=val_input_ids,
                                        attention_mask=val_attention_mask,
                                        labels=val_labels,
                                        use_cache=False
                                    )
                                    val_loss_item = val_outputs.loss
                                val_epoch_loss += val_loss_item.item()
                                num_val_batches += 1
                        
                        avg_val_loss = val_epoch_loss / num_val_batches if num_val_batches > 0 else float('inf')
                        logger.info(f"Step: {global_step}, Validation Loss: {avg_val_loss:.4f}")
                        if args.use_wandb:
                            # wandb.log({"val_loss": avg_val_loss, "global_step": global_step})
                            pass

                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-best")
                            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {checkpoint_dir}")
                        else:
                            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            logger.info(f"Saving checkpoint to {checkpoint_dir} (val loss {avg_val_loss:.4f} did not improve from {best_val_loss:.4f})")
                    else: # No validation data
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        logger.info(f"No validation data. Saving checkpoint to {checkpoint_dir}")

                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    # No tokenizer to save for this custom data, but save config
                    with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
                        json.dump(vars(args), f, indent=4)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                    model.train() # Switch back to train mode
            
            if global_step >= max_train_steps:
                break
        
        avg_epoch_train_loss = train_epoch_loss / num_train_batches if num_train_batches > 0 else 0
        epoch_end_time = time.time()
        logger.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds. Avg Train Loss: {avg_epoch_train_loss:.4f}")
        if args.use_wandb:
            # wandb.log({"epoch_train_loss": avg_epoch_train_loss, "epoch": epoch + 1})
            pass
        
        if global_step >= max_train_steps:
            logger.info("Maximum training steps reached. Exiting training.")
            break

    # Save final model
    final_checkpoint_dir = os.path.join(args.output_dir, "final_model")
    Path(final_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_checkpoint_dir)
    with open(os.path.join(final_checkpoint_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved final model to {final_checkpoint_dir}")

    if args.use_wandb:
        # wandb.finish()
        pass

    logger.info("Training finished.")

if __name__ == "__main__":
    train()
