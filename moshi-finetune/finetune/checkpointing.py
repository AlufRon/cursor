import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
import torch
from moshi.models.lm import LMModel
from moshi.modules.lora import LoRALinear
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states with complete training metadata"""

    def __init__(
        self,
        model: FullyShardedDataParallel | LMModel,
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        full_finetuning: bool = False,
        training_args=None,  # Add training args to capture complete config
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep
        self.full_finetuning = full_finetuning
        self.config = config
        self.training_args = training_args  # Store for metadata extraction

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(ckpt_dir: Path, save_only_lora: bool = False) -> Path:
        suffix = "safetensors"
        prefix = "lora" if save_only_lora else "consolidated"

        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    def extract_lora_config(self) -> dict:
        """Extract LoRA configuration from the model"""
        lora_config = {"enabled": False}
        
        # Find any LoRA layer to extract config
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                lora_config = {
                    "enabled": True,
                    "rank": module.rank,
                    "scaling": module.scaling,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
                break
        
        return lora_config

    def extract_ttt_config(self) -> dict:
        """Extract TTT configuration from the model"""
        ttt_config = {"enabled": False}
        
        # Get the actual model (unwrap FSDP if needed)
        actual_model = self.model
        if isinstance(self.model, FullyShardedDataParallel):
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Extract TTT config if TTT is enabled
        if hasattr(actual_model, 'user_ttt_model') and actual_model.user_ttt_model is not None:
            # Get the raw TTT config
            raw_config = actual_model.user_ttt_config.to_dict() if hasattr(actual_model.user_ttt_config, 'to_dict') else vars(actual_model.user_ttt_config)
            
            # Convert parameter names to model-expected format
            model_config = {}
            for key, value in raw_config.items():
                if key == "hidden_size":
                    model_config["ttt_hidden_size"] = value
                elif key == "intermediate_size":
                    model_config["ttt_intermediate_size"] = value
                elif key == "num_hidden_layers":
                    model_config["ttt_num_hidden_layers"] = value
                elif key == "num_attention_heads":
                    model_config["ttt_num_attention_heads"] = value
                elif key == "max_position_embeddings":
                    model_config["ttt_max_position_embeddings"] = value
                elif key == "layer_type":
                    model_config["ttt_layer_type"] = value
                elif key == "pre_conv":
                    model_config["ttt_pre_conv"] = value
                elif key == "conv_kernel":
                    model_config["ttt_conv_kernel"] = value
                elif key == "use_gate":
                    model_config["ttt_use_gate"] = value
                elif key == "scan_checkpoint_group_size":
                    model_config["ttt_scan_checkpoint_group_size"] = value
                else:
                    # Keep other parameters as-is
                    model_config[key] = value
            
            ttt_config = {
                "enabled": True,
                "config": model_config,
                "integration_weight": getattr(actual_model, 'ttt_integration_weight', 0.5),
                "has_input_projection": hasattr(actual_model, 'ttt_input_projection'),
                "has_output_projection": hasattr(actual_model, 'ttt_output_projection'),
                "has_projection": hasattr(actual_model, 'ttt_projection'),
            }
            
            # Add projection layer dimensions if they exist
            if hasattr(actual_model, 'ttt_input_projection'):
                ttt_config["input_projection_dims"] = {
                    "in_features": actual_model.ttt_input_projection.in_features,
                    "out_features": actual_model.ttt_input_projection.out_features,
                }
            
            if hasattr(actual_model, 'ttt_output_projection'):
                ttt_config["output_projection_dims"] = {
                    "in_features": actual_model.ttt_output_projection.in_features,
                    "out_features": actual_model.ttt_output_projection.out_features,
                }
                
            if hasattr(actual_model, 'ttt_projection'):
                ttt_config["projection_dims"] = {
                    "in_features": actual_model.ttt_projection.in_features,
                    "out_features": actual_model.ttt_projection.out_features,
                }
        
        return ttt_config

    def extract_model_architecture_config(self) -> dict:
        """Extract complete model architecture configuration"""
        actual_model = self.model
        if isinstance(self.model, FullyShardedDataParallel):
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        architecture_config = {
            "model_class": actual_model.__class__.__name__,
            "dtype": str(actual_model.dtype) if hasattr(actual_model, 'dtype') else "unknown",
            "device": str(actual_model.device) if hasattr(actual_model, 'device') else "unknown",
            "dim": getattr(actual_model, 'dim', None),
            "num_heads": getattr(actual_model, 'num_heads', None),
            "num_layers": getattr(actual_model, 'num_layers', None),
            "text_card": getattr(actual_model, 'text_card', None),
            "card": getattr(actual_model, 'card', None),
            "n_q": getattr(actual_model, 'n_q', None),
            "dep_q": getattr(actual_model, 'dep_q', None),
        }
        
        return architecture_config

    def extract_parameter_shapes(self) -> dict:
        """Extract all parameter shapes for validation"""
        parameter_shapes = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only trainable parameters
                parameter_shapes[name] = list(param.shape)
        
        return parameter_shapes

    def create_training_metadata(self) -> dict:
        """Create complete training metadata for exact architecture reconstruction"""
        metadata = {
            "version": "1.0",
            "checkpoint_type": "lora_ttt" if not self.full_finetuning else "full_finetune",
            "training_step": self.state.step,
            
            # Architecture configurations
            "lora_config": self.extract_lora_config(),
            "ttt_config": self.extract_ttt_config(),
            "model_architecture": self.extract_model_architecture_config(),
            "parameter_shapes": self.extract_parameter_shapes(),
            
            # Training configurations
            "training_config": self.config,
            "full_finetuning": self.full_finetuning,
            
            # Training args if available
            "training_args": None,
        }
        
        # Add training args if available
        if self.training_args is not None:
            try:
                # Convert training args to dict (handle dataclass)
                if hasattr(self.training_args, '__dict__'):
                    training_args_dict = {}
                    for key, value in self.training_args.__dict__.items():
                        try:
                            # Handle nested dataclasses
                            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                                training_args_dict[key] = value.__dict__
                            else:
                                training_args_dict[key] = value
                        except:
                            training_args_dict[key] = str(value)
                    metadata["training_args"] = training_args_dict
            except Exception as e:
                logger.warning(f"Could not serialize training_args: {e}")
        
        return metadata

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "config.json"
        
        # Create enhanced config with complete metadata
        enhanced_config = dict(self.config)
        enhanced_config["checkpoint_metadata"] = self.create_training_metadata()
        
        with open(params_path, "w") as f:
            f.write(json.dumps(enhanced_config, indent=4))

    @staticmethod
    def get_non_lora_states(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    @torch.no_grad()
    def retrieve_save_states(
        self, save_only_lora: bool, save_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        assert not (save_only_lora and self.full_finetuning), (
            "Cannot save LoRA checkpoint as LoRA training is not enabled."
        )

        # remove all potential hooks
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()  # type: ignore

        offload_to_cpu = get_world_size() > 1
        if save_only_lora:

            def is_trainable_fsdp(module: torch.nn.Module | FullyShardedDataParallel):
                is_fsdp = (isinstance(module, FullyShardedDataParallel) or get_world_size() == 1)
                all_params_have_grads = is_fsdp and all(p.requires_grad for p in module.parameters())

                # need to make sure only lowest fsdp wrap is used
                is_leaf_node = is_fsdp and (
                    get_world_size() == 1 or len(list(module.module.children())) == 0
                )  # type: ignore

                return is_fsdp and all_params_have_grads and is_leaf_node

            # extract all modules with only trainable weights
            modules = {
                k: m for k, m in self.model.named_modules() if is_trainable_fsdp(m)
            }

            states = {}
            for key, module in modules.items():
                assert isinstance(module, FullyShardedDataParallel) or get_world_size() == 1, (
                    "`module` should be an instance of `FullyShardedDataParallel` if `world_size > 1`"
                )
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                if get_world_size() > 1:
                    with module.summon_full_params(
                        module, writeback=True, offload_to_cpu=offload_to_cpu
                    ):
                        states.update(
                            {
                                f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                                for k, v in module.state_dict().items()
                            }
                        )
                else:
                    states.update(
                        {
                            f"{parent_prefix}.{k}": v.clone().to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        }
                    )
        else:
            # merge weights if we don't just save LoRA
            def merge_lora(
                m: torch.nn.Module,
                destination: dict[str, torch.Tensor],
                prefix: str,
                *args,
            ):
                weight = m.merge_weight()  # type: ignore
                destination[prefix + "weight"] = weight

            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(
                        merge_lora
                    )

            # make sure you have enough CPU RAM available to save the full model
            assert isinstance(self.model, FullyShardedDataParallel) or get_world_size() == 1, (
                "`self.model` should be an instance of `FullyShardedDataParallel` if `world_size > 1`"
            )
            if get_world_size() > 1:
                with self.model.summon_full_params(
                    self.model, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    states = self.get_non_lora_states(self.model.state_dict())
                    states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
            else:
                states = self.get_non_lora_states(self.model.state_dict())
                states = {k: v.clone().to(dtype=save_dtype) for k, v in states.items()}

        states = dict(sorted(states.items()))
        return states

    @torch.no_grad()
    def save_checkpoint(
        self,
        save_only_lora: bool,
        dtype: torch.dtype = torch.float16,
    ):
        if self.full_finetuning:
            assert not save_only_lora, "Cannot save LoRA checkpoint in full finetuning"

        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(
            f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}"
        )

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            states: dict[str, torch.Tensor] = self.retrieve_save_states(
                save_only_lora, dtype
            )

        barrier()

        if self.rank == 0:
            # Create complete training metadata
            training_metadata = self.create_training_metadata()
            
            # Convert metadata to strings for safetensors compatibility
            metadata_strings = {}
            for key, value in training_metadata.items():
                if isinstance(value, dict):
                    metadata_strings[key] = json.dumps(value)
                else:
                    metadata_strings[key] = str(value)
            
            # Save checkpoint in tmp path with metadata
            safetensors.torch.save_file(
                states,
                self.consolidated_path(
                    tmp_dst, save_only_lora=save_only_lora
                ),
                metadata=metadata_strings,  # Include complete training metadata
            )
            
            # Also save metadata as separate JSON for easy inspection
            metadata_path = tmp_dst / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(training_metadata, f, indent=4)
            
            self.write_params_info(tmp_dst)
            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}"
            )
            logger.info(f"Saved complete training metadata for exact architecture reconstruction")

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
