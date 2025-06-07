# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub.errors import EntryNotFoundError
except ImportError:
    from huggingface_hub.utils import EntryNotFoundError  # pyright: ignore
from safetensors.torch import load_model, load_file
import sentencepiece
import torch
import typing as tp
from .compression import MimiModel
from ..conditioners import BaseConditioner, ConditionProvider, ConditionFuser
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer
from ..modules.lora import replace_all_linear_with_lora, replace_lora_with_linear
import logging

logger = logging.getLogger(__name__)


SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
MOSHI_NAME = "model.safetensors"
MOSHI_Q8_NAME = "model.q8.safetensors"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def hf_get(filename: str | Path, hf_repo: str | None = None) -> Path:
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename[5:].split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif hf_repo is not None:
        return Path(hf_hub_download(hf_repo, filename))
    else:
        return Path(filename)


@dataclass
class CheckpointInfo:
    """
    Contains the paths to each sub model, along with some extra configuration.

    Args:
        moshi_weights: path to the checkpoint for the Moshi LM.
        mimi_weights: path to the checkpoint for the Mimi audio tokenizer.
        tokenizer: path to the text tokenizer.
        lm_config: config for instantiating the LM model.
            Can be None if the original Moshi 7B config should be used.
        raw_config: raw config, including original keys not intended for the LM.
        model_type: indicate the intended use, should be `moshi` or `hibiki`.
        lora_weights: path to an optional checkpoint with lora weights.
        lm_gen_config: optional default params to use for generation with this model.
        tts_config: optional TTS specific configuration.
        model_id: optional dict containing tracability information on the model origin, in particular
            its signature and epoch.
    """

    moshi_weights: Path
    mimi_weights: Path
    tokenizer: Path
    lm_config: dict | None = None
    raw_config: dict | None = None
    model_type: str = "moshi"
    lora_weights: Path | None = None
    lm_gen_config: dict = field(default_factory=dict)
    tts_config: dict = field(default_factory=dict)
    model_id: dict = field(default_factory=dict)

    @staticmethod
    def from_hf_repo(
        hf_repo: str,
        moshi_weights: Path | str | None = None,
        mimi_weights: Path | str | None = None,
        tokenizer: Path | str | None = None,
        config_path: Path | str | None = None,
        lora_weights: Path | str | None = None,
    ) -> "CheckpointInfo":
        """Downloads the checkpoints from the given repo, along with its config.

        Extra overrides are possible for each of Moshi, Mimi, or the text tokenizer,
        which should be either a Path to a local file or a string representing a path
        to a local file or starting with `hf://` for pointing to a file in another repo.

        Finally, a `config_path` can be provided to override the config from the repository.
        """
        if config_path is None:
            try:
                config_path = hf_hub_download(hf_repo, "config.json")
            except EntryNotFoundError:
                # No config.json, which might indicate legacy repository.
                warnings.warn(
                    f"Repository {hf_repo} contains no config.json. "
                    "Assuming this is a Moshi 7B. Support for such repository "
                    "might be removed in the future."
                )
        if config_path is None:
            moshi_name = MOSHI_NAME
            mimi_name = MIMI_NAME
            tokenizer_name = TEXT_TOKENIZER_NAME
            lm_config = None
            raw_config = None
            model_type = "moshi"
            lm_gen_config = {}
            tts_config = {}
            model_id = {}
            lora_name = None
        else:
            raw_config = json.loads(Path(config_path).read_text())
            lm_config = dict(raw_config)
            moshi_name = lm_config.pop("moshi_name", MOSHI_NAME)
            mimi_name = lm_config.pop("mimi_name", MIMI_NAME)
            tokenizer_name = lm_config.pop("tokenizer_name", TEXT_TOKENIZER_NAME)
            lora_name = lm_config.pop("lora_name", None)
            model_type = lm_config.pop("model_type", "moshi")
            lm_gen_config = lm_config.pop("lm_gen_config", {})
            tts_config = lm_config.pop("tts_config", {})
            model_id = lm_config.pop("model_id", {})

        if moshi_weights is None:
            moshi_weights_final = hf_get(moshi_name, hf_repo)
        else:
            moshi_weights_final = hf_get(moshi_weights)

        if mimi_weights is None:
            mimi_weights_final = hf_get(mimi_name, hf_repo)
        else:
            mimi_weights_final = hf_get(mimi_weights)

        if tokenizer is None:
            tokenizer_final = hf_get(tokenizer_name, hf_repo)
        else:
            tokenizer_final = hf_get(tokenizer)

        if lora_weights is None and lora_name:
            lora_weights_final = hf_get(lora_name, hf_repo)
        elif lora_weights is not None:
            lora_weights_final = hf_get(lora_weights)
        else:
            lora_weights_final = None

        return CheckpointInfo(
            moshi_weights_final,
            mimi_weights_final,
            tokenizer_final,
            lm_config,
            raw_config,
            model_type,
            lora_weights_final,
            lm_gen_config=lm_gen_config,
            tts_config=tts_config,
            model_id=model_id,
        )

    def get_mimi(self, device: torch.device | str = "cpu") -> MimiModel:
        if self.lm_config is None:
            num_codebooks = 8
        else:
            num_codebooks = max(self.lm_config["dep_q"], self.lm_config["n_q"] - self.lm_config["dep_q"])
        return get_mimi(self.mimi_weights, num_codebooks=num_codebooks, device=device)

    def get_moshi(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        load_weight: bool = True,
        **kwargs,
    ) -> LMModel:
        # Enable enhanced reconstruction for trained checkpoints by default
        # Check if this looks like a trained checkpoint (contains our metadata)
        use_exact_reconstruction = False
        if load_weight and self.moshi_weights:
            checkpoint_path = str(self.moshi_weights).lower()
            # Enable for checkpoints that likely contain our training metadata
            if any(indicator in checkpoint_path for indicator in ['lora', 'ttt', 'checkpoint', 'finetune', 'trained']):
                use_exact_reconstruction = True
        
        # Allow override from kwargs
        use_exact_reconstruction = kwargs.pop('use_exact_reconstruction', use_exact_reconstruction)
        
        model = get_moshi_lm(
            self.moshi_weights if load_weight else None,
            lm_kwargs=self.lm_config,
            device=device,
            dtype=dtype,
            lora_weights=self.lora_weights,
            use_exact_reconstruction=use_exact_reconstruction,
            **kwargs,
        )
        if self.model_type == "hibiki":
            # Sometime the model samples the EOS (2) too early, which we want to ignore.
            # We keep generating if the input file is not finished, and this is a way
            # to implicitely replace early EOS with PAD.
            model.text_emb.weight.data[2] = model.text_emb.weight.data[3]
        return model

    def get_text_tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        return sentencepiece.SentencePieceProcessor(str(self.tokenizer))  # type: ignore


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(
    filename: str | Path | None, device: torch.device | str = "cpu", num_codebooks: int = 8
) -> MimiModel:
    """Return a pretrained Mimi model, or unintialized if `filename` is None."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if filename is not None:
        if _is_safetensors(filename):
            load_model(model, filename, device=str(device))
        else:
            pkg = torch.load(filename, "cpu")
            model.load_state_dict(pkg["model"])
    model.set_num_codebooks(num_codebooks)
    return model


def load_training_metadata(filename: str | Path) -> tuple[dict, dict]:
    """Load complete training metadata from checkpoint"""
    metadata = {}
    state_dict = {}
    
    if _is_safetensors(filename):
        # Load from safetensors metadata
        import safetensors
        with safetensors.safe_open(filename, framework="pt") as f:
            # Load metadata
            for key in f.metadata():
                value = f.metadata()[key]
                try:
                    # Try to parse as JSON
                    metadata[key] = json.loads(value)
                except:
                    metadata[key] = value
            
            # Load state dict
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        # Load from pytorch checkpoint
        pkg = torch.load(filename, "cpu")
        if "fsdp_best_state" in pkg:
            state_dict = pkg["fsdp_best_state"]["model"]
        elif "model" in pkg:
            state_dict = pkg["model"]
        else:
            state_dict = pkg
            
        # Try to find metadata in the package
        metadata = pkg.get("metadata", {})
    
    return metadata, state_dict


def validate_architecture_compatibility(model, training_metadata: dict) -> bool:
    """Validate that the model architecture matches training metadata"""
    try:
        if "lora_config" in training_metadata:
            lora_config = training_metadata["lora_config"]
            if lora_config.get("enabled", False):
                # Check LoRA parameters exist
                lora_params = [name for name, _ in model.named_parameters() if "lora" in name]
                if not lora_params:
                    logger.error("Training used LoRA but loaded model has no LoRA parameters")
                    return False
        
        if "ttt_config" in training_metadata:
            ttt_config = training_metadata["ttt_config"]
            if ttt_config.get("enabled", False):
                # Check TTT model exists
                if not (hasattr(model, 'user_ttt_model') and model.user_ttt_model is not None):
                    logger.error("Training used TTT but loaded model has no TTT components")
                    return False
        
        if "parameter_shapes" in training_metadata:
            expected_shapes = training_metadata["parameter_shapes"]
            for name, param in model.named_parameters():
                if name in expected_shapes:
                    expected_shape = expected_shapes[name]
                    if list(param.shape) != expected_shape:
                        logger.error(f"Parameter shape mismatch: {name} expected {expected_shape}, got {list(param.shape)}")
                        return False
        
        return True
    except Exception as e:
        logger.error(f"Architecture validation failed: {e}")
        return False


def reconstruct_exact_training_architecture(
    training_metadata: dict,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lm_kwargs: dict = None,
    lm_kwargs_overrides: dict = None,
) -> tuple[LMModel, dict]:
    """
    Reconstruct the exact model architecture used during training.
    Returns: (model, lora_config_to_apply)
    """
    
    logger.info("🔧 Reconstructing exact training architecture from metadata...")
    
    # Start with base config
    if lm_kwargs is None:
        lm_kwargs = _lm_kwargs.copy()
    else:
        lm_kwargs = lm_kwargs.copy()
    
    # Apply any overrides
    if lm_kwargs_overrides:
        lm_kwargs.update(lm_kwargs_overrides)
    
    # Extract configurations from training metadata
    ttt_config = training_metadata.get("ttt_config", {})
    lora_config = training_metadata.get("lora_config", {})
    model_arch = training_metadata.get("model_architecture", {})
    training_args = training_metadata.get("training_args", {})
    
    # Apply TTT configuration if it was used during training
    if ttt_config.get("enabled", False):
        logger.info("✓ Enabling TTT with exact training configuration")
        
        # Use the exact TTT config from training
        ttt_model_config = ttt_config.get("config", {})
        
        # Filter TTT parameters - only pass model architecture parameters, not training parameters
        model_ttt_params = {
            "use_ttt": True,
            "ttt_integration_weight": ttt_config.get("integration_weight", 0.5),
        }
        
        # Add TTT model configuration parameters that are for model architecture
        architecture_ttt_params = [
            "ttt_hidden_size", "ttt_layer_type", "ttt_intermediate_size", 
            "ttt_num_hidden_layers", "ttt_num_attention_heads", 
            "ttt_max_position_embeddings", "ttt_pre_conv", "ttt_conv_kernel", 
            "ttt_use_gate", "ttt_scan_checkpoint_group_size"
        ]
        
        for key, value in ttt_model_config.items():
            if key in architecture_ttt_params:
                model_ttt_params[key] = value
            elif key.startswith("ttt_") and key not in ["ttt_base_lr", "mini_batch_size"]:
                # Include TTT architecture params but exclude training params
                model_ttt_params[key] = value
        
        # Apply TTT parameters to model kwargs
        lm_kwargs.update(model_ttt_params)
        
        logger.info(f"🔧 TTT model params: {list(model_ttt_params.keys())}")
    
    # Create model with exact training configuration (WITHOUT LoRA - that's applied later)
    logger.info(f"🏗️ Creating LMModel with exact training config...")
    model = LMModel(
        device=device,
        dtype=dtype,
        **lm_kwargs
    )
    
    # Return model and LoRA config to be applied separately
    lora_config_to_apply = None
    if lora_config.get("enabled", False):
        logger.info("✓ LoRA will be applied after model creation")
        lora_config_to_apply = lora_config
    
    return model, lora_config_to_apply


def get_moshi_lm_with_exact_reconstruction(
    filename: str | Path | None,
    lm_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_weights: str | Path | None = None,
    fuse_lora: bool = False,
    lm_kwargs_overrides={},
) -> LMModel:
    """Load model with exact training architecture reconstruction"""
    
    if lm_kwargs is None:
        lm_kwargs = _lm_kwargs.copy()
    
    # Apply overrides
    lm_kwargs = lm_kwargs | lm_kwargs_overrides
    
    # Handle deprecated params
    lm_kwargs.pop("depformer_causal", None)
    
    if filename is None:
        # No checkpoint, create fresh model
        return LMModel(device=device, dtype=dtype, **lm_kwargs)
    
    logger.info(f"🔍 Loading checkpoint with exact architecture reconstruction: {filename}")
    
    # Load training metadata and weights
    training_metadata, state_dict = load_training_metadata(filename)
    
    if not training_metadata:
        logger.warning("⚠️ No training metadata found, falling back to legacy loading")
        # Fall back to the old method if no metadata
        return get_moshi_lm_legacy(filename, lm_kwargs, device, dtype, lora_weights, fuse_lora, lm_kwargs_overrides)
    
    logger.info(f"✓ Found training metadata version: {training_metadata.get('version', 'unknown')}")
    
    # Debug: Show what metadata we have
    logger.info(f"📊 Available metadata keys: {list(training_metadata.keys())}")
    
    try:
        # Reconstruct exact training architecture
        model, lora_config_to_apply = reconstruct_exact_training_architecture(
            training_metadata=training_metadata,
            device=device,
            dtype=dtype,
            lm_kwargs=lm_kwargs,
            lm_kwargs_overrides=lm_kwargs_overrides,
        )
        
        # Apply LoRA after model creation if needed
        if lora_config_to_apply:
            logger.info("✓ Applying LoRA with exact training configuration")
            from ..modules.lora import replace_all_linear_with_lora
            replace_all_linear_with_lora(
                model,
                rank=lora_config_to_apply["rank"],
                scaling=lora_config_to_apply["scaling"],
                device=device,
                dtype=dtype
            )
        
        # Validate architecture compatibility
        if not validate_architecture_compatibility(model, training_metadata):
            logger.warning("Architecture validation failed, falling back to legacy method")
            return get_moshi_lm_legacy(filename, lm_kwargs, device, dtype, lora_weights, fuse_lora, lm_kwargs_overrides)
        
        # CRITICAL: Load base model weights first (if we don't have them in checkpoint)
        has_base_weights = any(key.startswith(('transformer.', 'text_emb.', 'depformer.')) and 
                              '.frozen_W.weight' in key for key in state_dict.keys())
        
        if not has_base_weights:
            logger.info("🔄 Loading base model weights first (checkpoint contains only adapters)...")
            # Determine base model path from training metadata or use default
            base_model_info = training_metadata.get("base_model", {})
            if "moshi_weights" in base_model_info:
                base_model_path = base_model_info["moshi_weights"]
                logger.info(f"📁 Using base model from metadata: {base_model_path}")
            else:
                                 # Use default base model - load it through the normal mechanism
                 logger.info("📁 Using default base model (moshiko-pytorch-bf16)")
                 try:
                     base_checkpoint = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
                     base_model_path = base_checkpoint.moshi_weights
                 except Exception as e:
                     logger.warning(f"Could not load default base model: {e}")
                     # Try with None to create fresh model
                     base_model_path = None
            
            if base_model_path:
                try:
                    # Load base model weights
                    if _is_safetensors(base_model_path):
                        from safetensors import safe_open
                        base_state_dict = {}
                        with safe_open(base_model_path, framework="pt") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)
                    else:
                        base_pkg = torch.load(base_model_path, "cpu")
                        if "model" in base_pkg:
                            base_state_dict = base_pkg["model"]
                        else:
                            base_state_dict = base_pkg
                    
                    # Convert base weights to target dtype
                    processed_base_dict = {}
                    for key, value in base_state_dict.items():
                        if value.dtype.is_floating_point:
                            if key.startswith('condition_provider.') or key.startswith('fuser.'):
                                processed_base_dict[key] = value.float()
                            else:
                                processed_base_dict[key] = value.to(dtype)
                        else:
                            processed_base_dict[key] = value
                    
                    # Load base weights first
                    logger.info(f"📥 Loading {len(processed_base_dict)} base model parameters...")
                    missing_base, unexpected_base = model.load_state_dict(processed_base_dict, strict=False, assign=True)
                    logger.info(f"✓ Base model loaded: {len(missing_base)} missing, {len(unexpected_base)} unexpected")
                    
                except Exception as e:
                    logger.warning(f"Failed to load base model weights: {e}")
        else:
            logger.info("✓ Checkpoint contains base model weights, no need to load separately")
        
        # Now load the checkpoint weights (adapters + any base weights in checkpoint)
        logger.info("🔄 Loading checkpoint weights (adapters)...")
        processed_state_dict = {}
        for key, value in state_dict.items():
            if value.dtype.is_floating_point:
                if key.startswith('condition_provider.') or key.startswith('fuser.'):
                    processed_state_dict[key] = value.float()
                else:
                    processed_state_dict[key] = value.to(dtype)
            else:
                processed_state_dict[key] = value
        
        # Load checkpoint weights (will override base weights if present, add adapters if not)
        missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False, assign=True)
        
        if missing_keys:
            logger.warning(f"Missing keys during checkpoint loading: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during checkpoint loading: {unexpected_keys}")
        
        # Final validation
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"   Total parameters: {param_count:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        
        # Log component verification
        if hasattr(model, 'user_ttt_model') and model.user_ttt_model is not None:
            ttt_params = sum(p.numel() for p in model.user_ttt_model.parameters())
            logger.info(f"   TTT model parameters: {ttt_params:,}")
        
        lora_params = sum(1 for name, _ in model.named_parameters() if "lora" in name)
        if lora_params > 0:
            logger.info(f"   LoRA parameters found: {lora_params}")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"❌ Exact reconstruction failed: {e}")
        logger.warning("Falling back to legacy loading method")
        return get_moshi_lm_legacy(filename, lm_kwargs, device, dtype, lora_weights, fuse_lora, lm_kwargs_overrides)


def get_moshi_lm_legacy(
    filename: str | Path | None,
    lm_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_weights: str | Path | None = None,
    fuse_lora: bool = False,
    lm_kwargs_overrides={},
) -> LMModel:
    """Legacy loading method for checkpoints without metadata (kept for backward compatibility)"""
    
    logger.warning("🔄 Using legacy loading method - this may cause architecture mismatches!")
    
    if lm_kwargs is None:
        lm_kwargs = _lm_kwargs
    lm_kwargs = dict(lm_kwargs)
    assert lm_kwargs is not None

    if "conditioners" in lm_kwargs:
        lm_kwargs["condition_provider"] = get_conditioner_provider(
            lm_kwargs["dim"], device, lm_kwargs
        )
        del lm_kwargs["conditioners"]
    if "fuser" in lm_kwargs:
        lm_kwargs["fuser"] = get_condition_fuser(lm_kwargs)

    lm_kwargs = lm_kwargs | lm_kwargs_overrides
    assert lm_kwargs is not None

    # deprecated params.
    lm_kwargs.pop("depformer_causal", None)

    # lora params.
    lora = lm_kwargs.pop("lora", False)
    lora_rank = lm_kwargs.pop("lora_rank", 128)
    lora_scaling = lm_kwargs.pop("lora_scaling", 2.0)

    # Pre-check for LoRA and TTT weights to enable them during model creation
    pre_lora_check = False
    pre_ttt_check = False
    detected_lora_rank = lora_rank  # Default rank
    detected_lora_scaling = lora_scaling  # Default scaling
    ttt_config = {}
    combined_checkpoint = False
    adapter_only_checkpoint = False
    
    # Try to load metadata first to get better parameters
    metadata_lora_rank = None
    metadata_lora_scaling = None
    if filename is not None:
        try:
            metadata, _ = load_training_metadata(filename)
            if metadata and "lora_config" in metadata:
                lora_config = metadata["lora_config"]
                if lora_config.get("enabled", False):
                    metadata_lora_rank = lora_config.get("rank")
                    metadata_lora_scaling = lora_config.get("scaling")
                    logger.info(f"📋 Found LoRA metadata: rank={metadata_lora_rank}, scaling={metadata_lora_scaling}")
        except Exception as e:
            logger.warning(f"Could not load metadata for legacy method: {e}")
    
    def extract_ttt_config_from_weights(state_dict):
        """Infer TTT configuration from checkpoint weights."""
        ttt_config = {}
        
        # Check if TTT weights exist
        ttt_keys = [k for k in state_dict.keys() if 'ttt' in k.lower() or 'user_ttt' in k]
        if not ttt_keys:
            return {}
            
        print(f"🧠 Found {len(ttt_keys)} TTT parameters, inferring configuration...")
        
        # Infer TTT configuration from weight shapes and names
        ttt_config["use_ttt"] = True
        
        # Find input projection to get hidden size
        for key, value in state_dict.items():
            if 'ttt_input_projection.weight' in key:
                ttt_config["ttt_hidden_size"] = value.shape[0]  # Output dimension
                break
        
        # Find output projection to verify
        for key, value in state_dict.items():
            if 'ttt_output_projection.weight' in key:
                if "ttt_hidden_size" not in ttt_config:
                    ttt_config["ttt_hidden_size"] = value.shape[1]  # Input dimension
                break
                
        # Set reasonable defaults based on common configurations
        ttt_config.update({
            "ttt_layer_type": "linear",  # Most common
            "ttt_integration_weight": 0.5,
            "ttt_intermediate_size": ttt_config.get("ttt_hidden_size", 1024) * 4,
            "ttt_num_hidden_layers": 2,
            "ttt_num_attention_heads": 8,
            "ttt_max_position_embeddings": 2048,
            "ttt_pre_conv": False,
            "ttt_conv_kernel": 3,
            "ttt_use_gate": True,
            "ttt_scan_checkpoint_group_size": 0,
        })
        
        print(f"🔧 Inferred TTT config: hidden_size={ttt_config.get('ttt_hidden_size')}")
        return ttt_config
    
    if filename is not None:
        # Quick check to see if we need LoRA/TTT enabled during model creation
        if _is_safetensors(filename):
            state = load_file(filename, device=str(device))
            pre_lora_check = any(key.startswith('lora_') or '.lora_A.' in key or '.lora_B.' in key for key in state.keys())
            pre_ttt_check = any('ttt' in key.lower() or 'user_ttt' in key for key in state.keys())
            
            # Use metadata rank if available, otherwise detect from weights
            if pre_lora_check:
                if metadata_lora_rank is not None:
                    detected_lora_rank = metadata_lora_rank
                    detected_lora_scaling = metadata_lora_scaling or lora_scaling
                    logger.info(f"🔍 Using LoRA rank from metadata: {detected_lora_rank}")
                else:
                    # Fallback to weight shape detection
                    for key, value in state.items():
                        if '.lora_A.' in key and len(value.shape) == 2:
                            detected_lora_rank = value.shape[0]  # First dimension is rank
                            logger.info(f"🔍 Detected LoRA rank from checkpoint weights: {detected_lora_rank}")
                            break
            
            # Extract TTT config from checkpoint if available
            if pre_ttt_check:
                ttt_config = extract_ttt_config_from_weights(state)
                
        else:
            pkg = torch.load(filename, "cpu")
            if "fsdp_best_state" in pkg:
                state_dict = pkg["fsdp_best_state"]["model"]
            elif "model" in pkg:
                state_dict = pkg["model"]
            else:
                state_dict = pkg
            pre_lora_check = any(key.startswith('lora_') or '.lora_A.' in key or '.lora_B.' in key for key in state_dict.keys())
            pre_ttt_check = any('ttt' in key.lower() or 'user_ttt' in key for key in state_dict.keys())
            
            # Use metadata rank if available, otherwise detect from weights
            if pre_lora_check:
                if metadata_lora_rank is not None:
                    detected_lora_rank = metadata_lora_rank
                    detected_lora_scaling = metadata_lora_scaling or lora_scaling
                    logger.info(f"🔍 Using LoRA rank from metadata: {detected_lora_rank}")
                else:
                    # Fallback to weight shape detection
                    for key, value in state_dict.items():
                        if '.lora_A.' in key and len(value.shape) == 2:
                            detected_lora_rank = value.shape[0]  # First dimension is rank
                            logger.info(f"🔍 Detected LoRA rank from checkpoint weights: {detected_lora_rank}")
                            break
            
            # Extract TTT config from checkpoint if available
            if pre_ttt_check:
                ttt_config = extract_ttt_config_from_weights(state_dict)
    
    # Enable LoRA if detected in checkpoint
    if pre_lora_check:
        lora = True
        lora_rank = detected_lora_rank  # Use the detected rank
        lora_scaling = detected_lora_scaling  # Use the detected scaling
        
    # Enable TTT if detected in checkpoint
    if pre_ttt_check:
        print(f"🧠 Enabling TTT during model creation...")
        lm_kwargs.update(ttt_config)  # Add TTT config to model kwargs

    # Initialize model on the target device directly
    model = LMModel(
        device=device,
        dtype=dtype,
        **lm_kwargs)

    # Set up LoRA layers if needed, before loading weights
    if pre_lora_check and not pre_ttt_check:
        # Only set up LoRA if TTT is not enabled (TTT model creation handles LoRA internally)
        print("🔧 Setting up LoRA layers before loading weights...")
        from ..modules.lora import replace_all_linear_with_lora
        replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=device)
    elif pre_lora_check and pre_ttt_check:
        print("🔧 LoRA layers handled by TTT-enabled model creation")

    # Now do the full checkpoint analysis and loading
    if filename is not None:
        if _is_safetensors(filename):
            # We already loaded this in pre-check, reuse it
            if not 'state' in locals():
                state = load_file(filename, device=str(device))
            # Check if this is a combined LoRA + TTT checkpoint
            has_lora_keys = any(key.startswith('lora_') or '.lora_A.' in key or '.lora_B.' in key for key in state.keys())
            has_ttt_keys = any('ttt' in key.lower() or 'user_ttt' in key for key in state.keys())
            # Base model keys are those that don't contain LoRA or TTT patterns
            has_base_keys = any(key.startswith(('transformer.', 'text_emb.', 'depformer.')) and 
                              not ('.lora_A.' in key or '.lora_B.' in key) and 
                              not ('ttt' in key.lower() or 'user_ttt' in key) 
                              for key in state.keys())
            
            if has_lora_keys or has_ttt_keys:
                if has_base_keys:
                    # This is a full checkpoint with base model weights - dangerous!
                    print(f"⚠️ WARNING: Checkpoint contains base model weights! This will override pretrained weights.")
                    combined_checkpoint = True
                else:
                    # This is adapter-only checkpoint (LoRA + TTT only) - safe
                    adapter_only_checkpoint = True
                    print(f"✅ Detected adapter-only checkpoint: LoRA={has_lora_keys}, TTT={has_ttt_keys}")
                    
            if combined_checkpoint or adapter_only_checkpoint:
                # Separate LoRA, TTT, and base model weights
                base_state = {}
                lora_state = {}
                ttt_state = {}
                
                for key, value in state.items():
                    if value.dtype.is_floating_point:
                        if key.startswith('condition_provider.') or key.startswith('fuser.'):
                            value = value.float()
                        else:
                            value = value.to(dtype)
                        state[key] = value
                    
                    # TTT classification takes priority (some TTT params have LoRA patterns)
                    if 'ttt' in key.lower() or 'user_ttt' in key:
                        ttt_state[key] = value
                    elif key.startswith('lora_') or '.lora_A.' in key or '.lora_B.' in key:
                        lora_state[key] = value
                    else:
                        base_state[key] = value
                
                print(f"📊 Checkpoint breakdown: {len(base_state)} base, {len(lora_state)} LoRA, {len(ttt_state)} TTT parameters")
                
                if adapter_only_checkpoint:
                    # Only load the adapter weights (LoRA + TTT), keep base model intact
                    adapter_state = {}
                    adapter_state.update(lora_state)
                    adapter_state.update(ttt_state)
                    print(f"🔄 Loading {len(adapter_state)} adapter parameters only")
                    model.load_state_dict(adapter_state, strict=False, assign=True)
                else:
                    # Load all weights together (dangerous - overwrites base model)
                    print(f"⚠️ Loading all {len(state)} parameters (including base model)")
                    model.load_state_dict(state, strict=False, assign=True)
                    
                lora = True  # Enable LoRA loading since we detected LoRA weights
            else:
                for key, value in state.items():
                    if value.dtype.is_floating_point:
                        if key.startswith('condition_provider.') or key.startswith('fuser.'):
                            value = value.float()
                        else:
                            value = value.to(dtype)
                        state[key] = value
                # Load state dict with assign=True to properly handle device placement
                model.load_state_dict(state, strict=False, assign=True)
        else:
            # We might have already loaded this in pre-check, reuse it
            if not 'pkg' in locals():
                pkg = torch.load(filename, "cpu",)
            # Check if this is a training checkpoint format
            if "fsdp_best_state" in pkg:
                state_dict = pkg["fsdp_best_state"]["model"]
            elif "model" in pkg:
                state_dict = pkg["model"]
            else:
                state_dict = pkg
                
            # Check for combined checkpoint
            has_lora_keys = any(key.startswith('lora_') or '.lora_A.' in key or '.lora_B.' in key for key in state_dict.keys())
            has_ttt_keys = any('ttt' in key.lower() or 'user_ttt' in key for key in state_dict.keys())
            has_base_keys = any(key.startswith(('transformer.', 'text_emb.', 'depformer.')) and 
                              not ('.lora_A.' in key or '.lora_B.' in key) and 
                              not ('ttt' in key.lower() or 'user_ttt' in key) 
                              for key in state_dict.keys())
            
            if has_lora_keys or has_ttt_keys:
                if has_base_keys:
                    print(f"⚠️ WARNING: Training checkpoint contains base model weights! This will override pretrained weights.")
                    combined_checkpoint = True
                else:
                    print(f"✅ Detected adapter-only training checkpoint: LoRA={has_lora_keys}, TTT={has_ttt_keys}")
                    adapter_only_checkpoint = True
                lora = True  # Enable LoRA loading
            
            if adapter_only_checkpoint:
                # Only load adapter weights (TTT + LoRA)
                adapter_state = {k: v for k, v in state_dict.items() 
                               if 'ttt' in k.lower() or 'user_ttt' in k or k.startswith('lora_') or '.lora_A.' in k or '.lora_B.' in k}
                print(f"🔄 Loading {len(adapter_state)} adapter parameters from training checkpoint")
                model.load_state_dict(adapter_state, strict=False, assign=True)
            else:
                # Load full state dict (may override base model if contains base weights)
                model.load_state_dict(state_dict, strict=False, assign=True)

    if lora:
        assert not lm_kwargs.get("quantize"), (
            "LoRA and quantization are incompatible for now."
        )
        if combined_checkpoint or adapter_only_checkpoint:
            # LoRA layers are already created and weights loaded
            if pre_ttt_check:
                print("🔄 LoRA+TTT model loaded successfully with all weights")
            else:
                print("🔄 LoRA layers already created and weights loaded from checkpoint")
            
            if fuse_lora:
                print("🔗 Fusing LoRA weights into base model...")
                from ..modules.lora import replace_lora_with_linear
                replace_lora_with_linear(model)
        else:
            # Use separate LoRA weights file
            model = get_lora_moshi(
                model=model,
                lora_rank=lora_rank,
                lora_scaling=lora_scaling,
                lora_weights=lora_weights,
                device=device,
                dtype=dtype,
                fuse_lora=fuse_lora,
            )
    else:
        assert lora_weights is None, (
            "`lora` is False, but received some lora_weights to load."
        )
    model.eval()
    return model


def get_moshi_lm(
    filename: str | Path | None,
    lm_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_weights: str | Path | None = None,
    fuse_lora: bool = False,
    lm_kwargs_overrides={},
    use_exact_reconstruction: bool = False,  # Default to False for training safety
) -> LMModel:
    """Load Moshi LM model with optional exact architecture reconstruction"""
    
    # Skip exact reconstruction for fresh model creation (training)
    if filename is None:
        use_exact_reconstruction = False
        
    # Skip exact reconstruction for base model files (training from pretrained)
    if filename is not None and isinstance(filename, (str, Path)):
        filename_str = str(filename).lower()
        # Skip for base model files that don't contain training metadata
        if any(name in filename_str for name in ['moshiko', 'moshi_weights', 'pytorch_model', 'model.safetensors']):
            if 'lora' not in filename_str and 'ttt' not in filename_str:
                use_exact_reconstruction = False
    
    if use_exact_reconstruction and filename is not None:
        try:
            return get_moshi_lm_with_exact_reconstruction(
                filename=filename,
                lm_kwargs=lm_kwargs,
                device=device,
                dtype=dtype,
                lora_weights=lora_weights,
                fuse_lora=fuse_lora,
                lm_kwargs_overrides=lm_kwargs_overrides,
            )
        except Exception as e:
            logger.warning(f"Exact reconstruction failed: {e}, falling back to legacy method")
            
    return get_moshi_lm_legacy(
        filename=filename,
        lm_kwargs=lm_kwargs,
        device=device,
        dtype=dtype,
        lora_weights=lora_weights,
        fuse_lora=fuse_lora,
        lm_kwargs_overrides=lm_kwargs_overrides,
    )


def get_conditioner(
    output_dim: int, device: torch.device | str, conditioner_cfg: dict
) -> BaseConditioner:
    conditioner_type = conditioner_cfg["type"]
    conditioner_kwargs = conditioner_cfg[conditioner_type]
    conditioner_kwargs.update({"output_dim": output_dim, "device": device})
    if conditioner_type == "lut":
        from ..conditioners.text import LUTConditioner
        return LUTConditioner(**conditioner_kwargs)
    elif conditioner_type == "tensor":
        from ..conditioners.tensors import TensorConditioner
        return TensorConditioner(**conditioner_kwargs)
    else:
        raise RuntimeError(f"Unknow conditioner type {conditioner_type}.")


def get_conditioner_provider(
    output_dim: int, device: torch.device | str, cfg: dict
) -> ConditionProvider:
    """Instantiate a conditioning model."""
    conditioners: tp.Dict[str, BaseConditioner] = {}
    for cond, cond_cfg in cfg["conditioners"].items():
        conditioners[cond] = get_conditioner(output_dim, device, cond_cfg)
    conditioner = ConditionProvider(conditioners, device=device)
    return conditioner


def get_condition_fuser(cfg: dict) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = cfg["fuser"]
    fuser_methods = ["sum", "cross", "prepend"]
    fuse2cond = {k: fuser_cfg.get(k, []) for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_lora_moshi(
    model: LMModel,
    lora_weights: str | Path | None,
    lora_rank: int,
    lora_scaling: float,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    fuse_lora: bool = True,
) -> LMModel:
    init_device = device
    if lora_weights is not None:
        init_device = torch.device('meta')
    replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=init_device)
    if lora_weights is not None:
        assert _is_safetensors(lora_weights), "LoRA weights must be a safetensors file."
        lora_state_dict = load_file(lora_weights, device=str(device))
        for key, value in lora_state_dict.items():
            if value.dtype.is_floating_point:
                value = value.to(dtype=dtype)
            lora_state_dict[key] = value
        res = model.load_state_dict(lora_state_dict, strict=False, assign=True)
        if res.unexpected_keys:
            raise RuntimeError(
                f"unexpected_keys in the lora weights: {res.unexpected_keys}"
            )
        model = model.to(dtype=dtype, device=device)
        if fuse_lora:
            replace_lora_with_linear(model)
    return model
