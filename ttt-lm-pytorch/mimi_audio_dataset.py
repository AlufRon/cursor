# mimi_audio_dataset.py
import os
import glob
import pickle
from pathlib import Path
from itertools import chain
import logging

import torch
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MimiInterleavedDataset(Dataset):
    """
    PyTorch Dataset for loading pre-tokenized Mimi audio tokens with
    Time-Aligned Interleaving.
    """
    def __init__(self,
                 data_dir: str,
                 split: str, # "train", "validation", or "test"
                 max_length: int,
                 num_codebooks: int,
                 codebook_size: int,
                 cache_dir_base: str = None, # Base directory for caching
                 skip_initial_tokens: int = 0, # Number of initial *interleaved* tokens to skip from each file
                 validation_override_dir: str = None, # Optional: specific directory for validation data
                 ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.skip_initial_tokens = skip_initial_tokens # This is the number of *final* interleaved tokens to skip
        self.validation_override_dir = Path(validation_override_dir) if validation_override_dir else None

        self.effective_vocab_size = self.num_codebooks * self.codebook_size

        self.cache_dir = None
        if cache_dir_base:
            cache_name = (
                f"mimi_interleaved-{self.split}-L{self.max_length}-NC{self.num_codebooks}-CS{self.codebook_size}"
                f"-skip{self.skip_initial_tokens}"
            )
            # Sanitize cache_name if data_dir has complex characters for path
            sanitized_data_dir_name = "".join(c if c.isalnum() else "_" for c in str(self.data_dir.name))
            cache_name = f"{sanitized_data_dir_name}-{cache_name}"

            self.cache_dir = Path(cache_dir_base) / cache_name
            self.cache_file_path = self.cache_dir / "concat_tokens.npy"
            self.tokenizer_info_path = self.cache_dir / "tokenizer_info.pkl"
        
        self.concat_tokens = self._load_or_process_data()

        if len(self.concat_tokens) < self.max_length + 1:
            if len(self.concat_tokens) == 0:
                 raise ValueError(
                    f"No tokens found for split '{self.split}' in directory '{self.data_dir}'. "
                    f"Ensure .pt files are present and correctly processed."
                )
            logger.warning(
                f"Total concatenated tokens ({len(self.concat_tokens)}) for split '{self.split}' "
                f"is less than max_length+1 ({self.max_length + 1}). "
                f"This dataset will be empty or very small."
            )


    def _find_token_files(self):
        """Finds .pt token files for the current split."""
        search_dir = self.data_dir
        
        if self.split == "validation" and self.validation_override_dir and self.validation_override_dir.is_dir():
            search_dir = self.validation_override_dir
            logger.info(f"Using validation override directory: {search_dir}")
        elif self.split == "validation" and self.validation_override_dir:
             logger.warning(f"Validation override directory {self.validation_override_dir} not found. Using default.")

        split_subdir = self.data_dir / self.split
        if split_subdir.is_dir():
            search_dir = split_subdir # Prefer split-specific subdirectory
            logger.info(f"Found specific subdirectory for split '{self.split}': {search_dir}")
        
        pt_files = sorted(glob.glob(str(search_dir / "**/*.pt"), recursive=True))
        
        if not pt_files and self.split == "train" and search_dir == self.data_dir:
            # Fallback for train if no 'train' subdir and search_dir was the main data_dir
             logger.info(f"No specific 'train' subdir, searching all .pt files in {self.data_dir} for training.")
             pt_files = sorted(glob.glob(str(self.data_dir / "**/*.pt"), recursive=True))


        logger.info(f"Found {len(pt_files)} tokenized .pt files for split '{self.split}' in {search_dir}.")
        if not pt_files:
            logger.warning(f"No .pt files found for split '{self.split}' in {search_dir} (and its subdirectories if applicable).")
        return pt_files

    def _process_single_file(self, file_path):
        """Loads a single .pt file and performs time-aligned interleaving."""
        try:
            # Expected shape: [batch_dim, num_codebooks, time_steps]
            # Typically batch_dim is 1 for these pre-tokenized files.
            tokens_tensor = torch.load(file_path, map_location='cpu')
            if tokens_tensor.ndim == 2: # If shape is [codebook, time], unsqueeze batch_dim
                tokens_tensor = tokens_tensor.unsqueeze(0)
            
            if tokens_tensor.size(0) != 1:
                logger.warning(f"File {file_path} has batch_dim {tokens_tensor.size(0)} > 1. Using only the first item.")
                tokens_tensor = tokens_tensor[0:1]

            if tokens_tensor.size(1) < self.num_codebooks:
                logger.warning(f"File {file_path} has {tokens_tensor.size(1)} codebooks, less than expected {self.num_codebooks}. Using available codebooks.")
            
            actual_num_codebooks = min(self.num_codebooks, tokens_tensor.size(1))
            time_steps = tokens_tensor.size(2)
            
            interleaved_tokens = []

            # Calculate how many time steps to skip based on interleaved tokens
            # E.g., if skip_initial_tokens = 8 (for 8 codebooks), it means skip 1 time step.
            time_steps_to_skip = 0
            if self.skip_initial_tokens > 0 and actual_num_codebooks > 0:
                time_steps_to_skip = self.skip_initial_tokens // actual_num_codebooks
            
            if time_steps <= time_steps_to_skip:
                # logger.debug(f"Skipping short file {Path(file_path).name} (time_steps: {time_steps}, to_skip: {time_steps_to_skip})")
                return []

            for t_idx in range(time_steps_to_skip, time_steps):
                for cb_idx in range(actual_num_codebooks):
                    token_val = tokens_tensor[0, cb_idx, t_idx].item()
                    offset = cb_idx * self.codebook_size
                    interleaved_tokens.append(token_val + offset)
            
            return interleaved_tokens

        except Exception as e:
            logger.error(f"Error loading or processing file {file_path}: {e}")
            return []

    def _load_or_process_data(self):
        """Loads data from cache or processes it if cache is not available/valid."""
        if self.cache_dir and self.cache_file_path.exists() and self.tokenizer_info_path.exists():
            try:
                logger.info(f"Loading processed data from cache: {self.cache_file_path}")
                concat_tokens = np.load(self.cache_file_path, mmap_mode='r')
                with open(self.tokenizer_info_path, "rb") as f:
                    tokenizer_info = pickle.load(f)
                
                # Validate cache (simple check)
                if (tokenizer_info.get("num_codebooks") == self.num_codebooks and
                    tokenizer_info.get("codebook_size") == self.codebook_size and
                    tokenizer_info.get("interleave_type") == "time_aligned_pytorch"):
                    logger.info(f"Cache loaded successfully for split '{self.split}'. Total tokens: {len(concat_tokens)}")
                    return concat_tokens
                else:
                    logger.info("Cache mismatch or outdated. Reprocessing...")
            except Exception as e:
                logger.warning(f"Could not load from cache ({e}). Reprocessing...")

        logger.info(f"Processing data for split '{self.split}' from directory: {self.data_dir}")
        
        token_files = self._find_token_files()
        if not token_files:
             # Return an empty array if no files are found, to be handled by __len__
            logger.warning(f"No token files found for split {self.split}. Dataset will be empty.")
            return np.array([], dtype=np.int32)

        all_tokens_list = []
        for i, file_path in enumerate(token_files):
            # if i % 100 == 0: # Log progress
            #     logger.info(f"Processing file {i+1}/{len(token_files)} for split '{self.split}'...")
            all_tokens_list.extend(self._process_single_file(file_path))
        
        if not all_tokens_list:
            logger.warning(f"No tokens were extracted after processing all files for split {self.split}. Dataset will be empty.")
            concat_tokens = np.array([], dtype=np.int32)
        else:
            concat_tokens = np.array(all_tokens_list, dtype=np.int32)
            logger.info(f"Finished processing for split '{self.split}'. Total interleaved tokens: {len(concat_tokens)}")


        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(self.cache_file_path, concat_tokens)
                tokenizer_info = {
                    "vocab_size": self.effective_vocab_size,
                    "num_codebooks": self.num_codebooks,
                    "codebook_size": self.codebook_size,
                    "interleave_type": "time_aligned_pytorch", # Mark as PyTorch processed
                    "data_source": str(self.data_dir),
                    "split": self.split
                }
                with open(self.tokenizer_info_path, "wb") as f:
                    pickle.dump(tokenizer_info, f)
                logger.info(f"Saved processed data to cache: {self.cache_file_path}")
            except Exception as e:
                logger.error(f"Failed to save data to cache: {e}")
        
        return concat_tokens

    def __len__(self):
        if len(self.concat_tokens) < self.max_length + 1:
            return 0 # Not enough tokens for even one sequence
        return (len(self.concat_tokens) - 1) // self.max_length

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError(f"Index must be an integer, got {type(idx)}")
        if idx < 0 or idx >= len(self):
             raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        start_idx = idx * self.max_length
        # LMDataset in JAX takes seq_len and creates sequences of seq_len+1
        # to get input_ids (0 to N-1) and labels (1 to N)
        end_idx = start_idx + self.max_length + 1
        
        token_chunk = self.concat_tokens[start_idx:end_idx]
        
        input_ids = torch.tensor(token_chunk[:-1], dtype=torch.long)
        labels = torch.tensor(token_chunk[1:], dtype=torch.long) # Shifted for next token prediction
        
        # For causal LM, attention mask is usually all ones for non-padded tokens
        # This dataset assumes no padding within the sequence itself.
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

