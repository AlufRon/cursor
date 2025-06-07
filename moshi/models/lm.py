# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import logging
import typing as tp
import torch
from torch import nn
from ..conditioners import ConditionProvider, ConditionFuser, ConditionTensors
from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..utils.quantize import replace_linear_with_qlinear
from ..modules.streaming import StreamingContainer, StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from .lm_utils import (_delay_sequence,
                       _undelay_sequence,
                       _init_layer,
                       ScaledEmbedding)
from .ttt import TTTModel as TTTModel, TTTConfig as TTTConfig, TTTCache
import math

logger = logging.getLogger(__name__)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


@dataclass
class LMModelState(State):
    """State for LMModel streaming.
    This is a minimal implementation that just stores a flag indicating
    the streaming state is initialized. The actual TTT cache is stored
    as a member variable in LMModel.
    """
    initialized: bool = True

    def reset(self, reset_mask: torch.Tensor) -> None:
        """Reset is handled by LMModel.reset_streaming which manages the TTT cache."""
        super().reset(reset_mask)
        # No additional reset needed here as LMModel.reset_streaming handles the TTT cache


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        depformer_weights_per_step_schedule (list[int] | None): mapping `CODEBOOK_INDEX -> WEIGHT_INDEX`, allowing
        depformer_low_rank_embeddings (int | None): if provided, uses low rank embeddings, with a linear
        existing_text_padding_id (int): token to use for the padding.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_low_rank_embeddings: int | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: int = 3,
        existing_text_end_padding_id: int = 0,
        context: tp.Optional[int] = None,
        causal: bool = True,
        condition_provider: tp.Optional[ConditionProvider] = None,
        fuser: tp.Optional[ConditionFuser] = None,
        quantize: bool = False,
        device=None,
        dtype=None,
        gradient_checkpointing: bool = False,
        ttt_integration_weight: float = 0.7,  # Weight for TTT in the combined output
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, f"expected {self.num_codebooks} delays, got {len(delays)}."
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.existing_text_end_padding_id = existing_text_end_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim)

        self.text_linear = nn.Linear(dim, text_card, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        
        # Extract TTT-related parameters from main_kwargs
        # (1) First check if TTT is enabled via 'use_ttt' param
        use_ttt = main_kwargs.pop('use_ttt', True)
        # (2) Extract other TTT-specific parameters
        ttt_integration_weight = main_kwargs.pop('ttt_integration_weight', 0.7)
        ttt_base_lr = main_kwargs.pop('ttt_base_lr', 1.0)
        ttt_mini_batch_size = main_kwargs.pop('ttt_mini_batch_size', 16)
        ttt_layer_type = main_kwargs.pop('ttt_layer_type', 'linear')
        
        # (3) Create a clean ttt_config dictionary with all TTT params
        ttt_config = None
        if use_ttt:
            logger.info(f"Initializing LMModel with TTT enabled (type: {ttt_layer_type})")
            ttt_config = {
                "mini_batch_size": ttt_mini_batch_size,
                "ttt_layer_type": ttt_layer_type,
                "ttt_base_lr": ttt_base_lr,
            }
            
            # (4) If user_ttt_config was already created, use that instead
            if hasattr(self, 'user_ttt_config'):
                logger.info("Using already initialized user_ttt_config")
                ttt_config = self.user_ttt_config.to_dict()
        
        # Create StreamingTransformer with properly filtered kwargs
        # Prepare transformer-compatible ttt_config
        transformer_ttt_config = None
        if ttt_config is not None:
            # Convert ttt_config to a simple dict without any implementation details
            # that might not be compatible with StreamingTransformer
            transformer_ttt_config = {
                "use_ttt": True,
                "ttt_layer_type": ttt_config.get("ttt_layer_type", "linear"),
                "hidden_size": 1024,
            }
            logger.info(f"Passing transformer_ttt_config to StreamingTransformer: {transformer_ttt_config}")
            
        # Remove ttt_config from main_kwargs if it exists to avoid duplicate
        if 'ttt_config' in main_kwargs:
            logger.info("Removing duplicate ttt_config from main_kwargs")
            del main_kwargs['ttt_config']
            
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            quantize=quantize,
            context=context,
            causal=causal,
            checkpointing=gradient_checkpointing,
            ttt_config=transformer_ttt_config,  # Pass the transformer-compatible ttt_config
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        kwargs_dep["cross_attention"] = False
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            num_in = dep_q
            if depformer_weights_per_step_schedule:
                num_in = max(depformer_weights_per_step_schedule) + 1
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(num_in)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        EmbeddingFactory = partial(EmbeddingFactory, low_rank=depformer_low_rank_embeddings)
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            weights_per_step_schedule=depformer_weights_per_step_schedule,
            causal=causal,
            quantize=quantize,
            checkpointing=gradient_checkpointing,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        # Depformer follow its own cycle of streaming entirely contained in one time step
        # and should not follow the streaming of the steps dimensions.
        self.depformer.set_streaming_detached(True)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )
        self.to(device=device, dtype=dtype)
        # We always keep the condition provider as float32.
        self.condition_provider = condition_provider
        self.fuser = fuser
        if self.condition_provider is not None:
            self.condition_provider.to(device=device)
        if self.fuser is not None:
            self.fuser.to(device=device)
        self._init_weights()
        if quantize:
            replace_linear_with_qlinear(self)

        # Initialize TTT components only if TTT is enabled
        if use_ttt:
            logger.info("Initializing TTT model components")
            
            # Determine correct device for TTT initialization
            # During finetuning with FSDP, device might be "meta" initially
            # Force CPU for the TTT components to avoid meta parameters
            init_device = device
            is_meta_device = (device is not None and str(device) == "meta")
            
            if is_meta_device:
                logger.warning("Meta device detected during TTT initialization. Forcing TTT components to be on CPU instead.")
                init_device = "cpu"
            else:
                logger.info(f"Initializing TTT components on device: {init_device}")
            ttt_hidden_size = dim 
                        # TTTModel config with proper parameters
            ttt_config_params = {
                "hidden_size": ttt_hidden_size,
                "intermediate_size": int(hidden_scale * ttt_hidden_size),
                "num_hidden_layers": 4,  # Fewer layers than original model
                "num_attention_heads": num_heads,
                "max_position_embeddings": context if context is not None else 2048,
                "vocab_size": text_card,
                "mini_batch_size": ttt_mini_batch_size,
                "ttt_layer_type": ttt_layer_type,
                "use_cache": True,
                "ttt_base_lr": ttt_base_lr,
            }
            # Import TTT modules here to ensure they're available
            try:
                from .ttt import TTTConfig, TTTModel
                
                self.user_ttt_config = TTTConfig(**ttt_config_params)
                
                # CRITICAL FIX: Force creation on CPU to avoid meta tensor issues
                with torch.device('cpu'):
                    self.user_ttt_model = TTTModel(self.user_ttt_config)
                    logger.info(f"Created TTTModel on CPU to avoid meta tensor issues")
                    
                    # Initialize all TTT parameters properly on CPU first
                    for name, param in self.user_ttt_model.named_parameters():
                        if param.is_meta:
                            # Replace meta parameters with actual tensors
                            if 'weight' in name:
                                param.data = torch.normal(0, 0.02, size=param.shape, dtype=dtype)
                            elif 'bias' in name:
                                param.data = torch.zeros(param.shape, dtype=dtype)
                            else:
                                param.data = torch.randn(param.shape, dtype=dtype) * 0.02
                            logger.info(f"Replaced meta tensor for {name}")
                        else:
                            # Ensure correct dtype
                            param.data = param.data.to(dtype=dtype)
                
                # Now move to target device if needed
                if not is_meta_device and str(init_device) != 'cpu':
                    self.user_ttt_model = self.user_ttt_model.to(device=init_device)
                    logger.info(f"Moved TTTModel to device: {init_device}")
                elif is_meta_device:
                    logger.info(f"Keeping TTTModel on CPU due to meta device context")
                
                # Convert TTT model to target dtype (redundant but safe)
                logger.info(f"Ensuring TTTModel has dtype: {dtype}")
                self.user_ttt_model = self.user_ttt_model.to(dtype=dtype)
                
                # Ensure use_cache is enabled for TTT model
                if hasattr(self.user_ttt_model, 'config'):
                    if not self.user_ttt_model.config.use_cache:
                        logger.info("LMModel: Forcing user_ttt_model.config.use_cache = True for manual cache management.")
                        self.user_ttt_model.config.use_cache = True
                
                # Add a member to hold the managed TTTCache
                self._managed_ttt_cache = None
                
                # Store the TTT integration weight
                self.ttt_integration_weight = ttt_integration_weight
                
                # Projection layer for TTT outputs - create on CPU first
                logger.info(f"Initializing TTT projection layer: {self.user_ttt_model.config.hidden_size} -> {dim}")
                with torch.device('cpu'):
                    self.ttt_projection = nn.Linear(self.user_ttt_model.config.hidden_size, dim, bias=True)
                    
                    # Initialize projection layer parameters properly
                    torch.nn.init.normal_(self.ttt_projection.weight, 0, 0.02)
                    torch.nn.init.zeros_(self.ttt_projection.bias)
                    
                    # Convert to target dtype
                    self.ttt_projection = self.ttt_projection.to(dtype=dtype)
                    logger.info(f"Created TTT projection layer on CPU with dtype: {dtype}")
                
                # Move to target device if needed
                if not is_meta_device and str(init_device) != 'cpu':
                    self.ttt_projection = self.ttt_projection.to(device=init_device)
                    logger.info(f"Moved TTT projection layer to device: {init_device}")
                elif is_meta_device:
                    logger.info(f"Keeping TTT projection layer on CPU due to meta device context")
                
                # Confirm these are not meta tensors and convert to target dtype
                for name, param in self.ttt_projection.named_parameters():
                    logger.info(f"TTT projection parameter '{name}' created on device {param.device}, is_meta={param.is_meta}, dtype={param.dtype}")
                
                # Verify no meta tensors remain
                logger.info("Verifying TTT parameter initialization...")
                meta_tensor_count = 0
                for name, param in self.named_parameters():
                    if ('ttt_projection' in name or 'user_ttt_model' in name) and param.is_meta:
                        logger.error(f"CRITICAL: Found remaining meta tensor: {name}")
                        meta_tensor_count += 1
                
                if meta_tensor_count > 0:
                    logger.error(f"Found {meta_tensor_count} meta tensors in TTT components!")
                    raise RuntimeError("TTT initialization failed - meta tensors remain")
                else:
                    logger.info("✅ All TTT parameters properly initialized (no meta tensors)")
                
                # Final verification that all TTT parameters have correct dtype
                logger.info("Verifying TTT parameter dtypes...")
                ttt_param_count = 0
                correct_dtype_count = 0
                for name, param in self.named_parameters():
                    if 'ttt_projection' in name or 'user_ttt_model' in name:
                        ttt_param_count += 1
                        if param.dtype == dtype:
                            correct_dtype_count += 1
                        else:
                            logger.warning(f"TTT parameter {name} has incorrect dtype {param.dtype}, expected {dtype}")
                            # Force convert to correct dtype
                            param.data = param.data.to(dtype)
                            correct_dtype_count += 1
                
                logger.info(f"TTT dtype verification: {correct_dtype_count}/{ttt_param_count} parameters have correct dtype {dtype}")
                logger.info("TTT components successfully initialized")
            except ImportError as e:
                logger.error(f"Failed to import TTT modules: {e}")
                self.user_ttt_model = None
                self.user_ttt_config = None
                self._managed_ttt_cache = None
                self.ttt_projection = None
                self.ttt_integration_weight = 0.0
            except Exception as e:
                logger.error(f"Failed to initialize TTT components: {e}")
                self.user_ttt_model = None
                self.user_ttt_config = None
                self._managed_ttt_cache = None
                self.ttt_projection = None
                self.ttt_integration_weight = 0.0
        else:
            logger.info("TTT is disabled - not initializing TTT model components")
            self.user_ttt_model = None
            self.user_ttt_config = None
            self._managed_ttt_cache = None
            self.ttt_projection = None
            self.ttt_integration_weight = 0.0

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return self.existing_text_end_padding_id

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self) -> torch.device:
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def dtype(self) -> torch.dtype:
        first_param = next(iter(self.text_emb.parameters()))
        return first_param.dtype

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward(
            self, codes: torch.Tensor,
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, returns the logits
        along with masks indicating the valid positions at which to compute the loss.
        The logits time steps are aligned with those in the input `code`.
        Should only be used for training, not inference (use `LMGen` for that).

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps. When text is supported,
                the first 'codebook' corresponds to the text, and the remaining codebooks are for the  audio.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning tensors.
        Returns:
            LMOutput: Language model outputs, containing either text or audio logits, or both.
                logits (torch.Tensor, or None) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor, or None) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
                text_logits (torch.Tensor, or None) of shape [B, 1, T, text_card].
                text_mask (torch.Tensor, or None) of shape [B, 1, T], mask over the valid positions for the text.
        """
        # Add explicit debug logging
        logger.info("LMModel.forward: Starting forward pass")
        try:
            B, K, T = codes.shape
            assert K == self.num_codebooks, (K, self.num_codebooks)
            # Delaying codes and removing the last time step that will never be an input.
            initial = self._get_initial_token().expand(B, -1, -1)
            delayed_codes = _delay_sequence(self.delays, codes, initial)
            # Inserting the empty tokens for the first time step.
            delayed_codes = torch.cat([initial, delayed_codes], dim=2)

            sum_condition: torch.Tensor | None = None
            cross_attention_src: torch.Tensor | None = None
            if condition_tensors is None:
                assert self.fuser is None
            else:
                assert self.fuser is not None
                sum_condition = self.fuser.get_sum(condition_tensors)
                cross_attention_src = self.fuser.get_cross(condition_tensors)

            transformer_out, text_logits = self.forward_text(delayed_codes[:, :, :-1], sum_condition, cross_attention_src)
            assert transformer_out.shape[0] == delayed_codes.shape[0]
            assert transformer_out.shape[1] == delayed_codes.shape[2] - 1
            logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

            # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
            # and provide the corresponding mask over invalid positions of tokens. We will with NaN values invalid positions
            # to ensure they properly handled.
            logits, logits_mask = _undelay_sequence(
                self.delays[self.audio_offset:self.audio_offset + self.dep_q],
                logits, fill_value=float('NaN'))
            logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
            text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
            text_logits_mask &= (codes[:, :1] != self.zero_token_id)
            
            # Create and validate LMOutput
            output = LMOutput(logits, logits_mask, text_logits, text_logits_mask)
            
            # Debug logging to verify the returned object
            logger.info(f"LMModel.forward: Created LMOutput with text_mask shape: {output.text_mask.shape}, "
                        f"text_logits shape: {output.text_logits.shape}, "
                        f"mask shape: {output.mask.shape}, "
                        f"logits shape: {output.logits.shape}")
                        
            # Final verification and type guarantee
            if not isinstance(output, LMOutput):
                logger.error(f"LMModel.forward: Critical error! Output is not LMOutput but {type(output)}")
                
            # Always create a fresh LMOutput to guarantee proper type
            # This ensures we're not returning some subclass or modified version
            final_output = LMOutput(
                logits=output.logits if hasattr(output, 'logits') else logits,
                mask=output.mask if hasattr(output, 'mask') else logits_mask,
                text_logits=output.text_logits if hasattr(output, 'text_logits') else text_logits,
                text_mask=output.text_mask if hasattr(output, 'text_mask') else text_logits_mask
            )
            
            logger.info(f"LMModel.forward: Returning fresh LMOutput instance of type {type(final_output)}")
            return final_output
            
        except Exception as e:
            logger.error(f"LMModel.forward: Exception occurred: {e}")
            # Get exception traceback for debugging
            import traceback
            logger.error(f"LMModel.forward: Traceback: {traceback.format_exc()}")
            
            # In case of any failure, create an empty LMOutput with the original input shape
            B, K, T = codes.shape
            device = codes.device
            # Create dummy tensors with the right shape
            dummy_logits = torch.zeros((B, self.dep_q, T, self.card), device=device)
            dummy_mask = torch.zeros((B, self.dep_q, T), dtype=torch.bool, device=device)
            dummy_text_logits = torch.zeros((B, 1, T, self.text_card), device=device)
            dummy_text_mask = torch.zeros((B, 1, T), dtype=torch.bool, device=device)
            
            fallback_output = LMOutput(dummy_logits, dummy_mask, dummy_text_logits, dummy_text_mask)
            logger.info("LMModel.forward: Created emergency fallback LMOutput due to exception")
            
            return fallback_output

    def forward_text(
        self,
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process text and audio embeddings through the transformer models."""
        logger.info(f"LMModel.forward_text: Starting forward_text with sequence shape {sequence.shape}")
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks {self.num_codebooks}."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])

        input_ = text_emb if input_ is None else input_ + text_emb
        if sum_condition is not None:
            input_ = input_ + sum_condition.to(input_)
        if cross_attention_src is not None:
            cross_attention_src = cross_attention_src.to(input_)
        
        # Run Moshi's original transformer
        logger.info(f"Running main transformer with input shape {input_.shape}")
        moshi_transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
        logger.info(f"Main transformer output shape: {moshi_transformer_out.shape}")
        
        # If TTT is enabled, run the TTT model and combine the outputs
        if hasattr(self, 'user_ttt_model') and self.user_ttt_model is not None:
            try:
                logger.info(f"Running TTT model with input shape {input_.shape}")
                
                # Create proper attention mask
                attention_mask = torch.ones((B, S), device=input_.device)
                
                # Create proper position IDs that respect current sequence position
                position_offset = 0
                
                # CRITICAL FIX: Disable TTT cache during training to avoid in-place operations that break gradients
                # Only use cache during inference (when not in training mode)
                ttt_cache_to_pass = None
                position_offset = 0
                
                if not self.training and self.is_streaming:
                    # Only use cache during inference, not training
                    if self._managed_ttt_cache is None:
                        logger.info(f"LMModel: Creating TTTCache for inference with batch_size: {B}")
                        self._managed_ttt_cache = TTTCache(self.user_ttt_model, B)
                    
                    ttt_cache_to_pass = self._managed_ttt_cache
                    position_offset = ttt_cache_to_pass.seqlen_offset
                    logger.debug(f"LMModel: Using managed TTTCache for inference with position_offset={position_offset}")
                else:
                    logger.debug(f"LMModel: Disabling TTT cache during training (training={self.training}, streaming={self.is_streaming})")
                
                position_ids = torch.arange(
                    position_offset,
                    position_offset + S,
                    dtype=torch.long,
                    device=input_.device
                ).unsqueeze(0).expand(B, -1)
                
                # Check and apply input projection if dimensions don't match
                input_needs_projection = input_.shape[-1] != self.user_ttt_model.config.hidden_size
                logger.info(f"TTT input projection needed: {input_needs_projection}, " 
                        f"input dim: {input_.shape[-1]}, TTT dim: {self.user_ttt_model.config.hidden_size}")
                
                if input_needs_projection:
                    # Add an input projection if dimensions don't match
                    if not hasattr(self, 'ttt_input_projection'):
                        logger.info(f"Creating input projection layer from {input_.shape[-1]} to {self.user_ttt_model.config.hidden_size}")
                        self.ttt_input_projection = nn.Linear(
                            input_.shape[-1], 
                            self.user_ttt_model.config.hidden_size,
                            device=input_.device,
                            dtype=input_.dtype  # Match the dtype of the input
                        )
                        # Initialize weights properly
                        nn.init.normal_(self.ttt_input_projection.weight, std=0.02)
                        nn.init.zeros_(self.ttt_input_projection.bias)
                    
                    # Ensure weight is the right dtype
                    if self.ttt_input_projection.weight.dtype != input_.dtype:
                        logger.info(f"Converting ttt_input_projection from {self.ttt_input_projection.weight.dtype} to {input_.dtype}")
                        self.ttt_input_projection = self.ttt_input_projection.to(dtype=input_.dtype)
                    
                    # Project input to match TTT model dimensions
                    ttt_input = self.ttt_input_projection(input_)
                    logger.info(f"Projected input shape: {ttt_input.shape}")
                else:
                    ttt_input = input_
                    logger.info(f"Using original input shape: {ttt_input.shape}")
                
                # Verify the input dimension matches TTT model's expected dimension
                assert ttt_input.shape[-1] == self.user_ttt_model.config.hidden_size, \
                    f"Input dimension {ttt_input.shape[-1]} doesn't match TTT model's expected dimension {self.user_ttt_model.config.hidden_size}"
                
                # Run TTT model - disable cache during training to avoid in-place gradient issues
                ttt_outputs = self.user_ttt_model(
                    inputs_embeds=ttt_input,  # Use projected input
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=ttt_cache_to_pass,  # None during training, cache during inference
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=not self.training and self.user_ttt_model.config.use_cache  # Disable cache during training
                )
                ttt_transformer_out = ttt_outputs.last_hidden_state
                logger.info(f"TTT transformer output shape: {ttt_transformer_out.shape}")
                
                # Log TTT state to wandb periodically (every 50 steps)
                if ttt_cache_to_pass is not None and hasattr(ttt_cache_to_pass, 'log_ttt_state_to_wandb'):
                    if ttt_cache_to_pass.seqlen_offset % 50 == 0:
                        try:
                            ttt_cache_to_pass.log_ttt_state_to_wandb()
                        except Exception as e:
                            logger.warning(f"Failed to log TTT state to wandb: {e}")
                
                # Handle output projection if needed
                output_needs_projection = ttt_transformer_out.shape[-1] != moshi_transformer_out.shape[-1]
                logger.info(f"TTT output projection needed: {output_needs_projection}, "
                        f"TTT output dim: {ttt_transformer_out.shape[-1]}, target dim: {moshi_transformer_out.shape[-1]}")
                
                if output_needs_projection:
                    # Create output projection if needed and not already created
                    if not hasattr(self, 'ttt_output_projection'):
                        logger.info(f"Creating output projection from {ttt_transformer_out.shape[-1]} to {moshi_transformer_out.shape[-1]}")
                        self.ttt_output_projection = nn.Linear(
                            ttt_transformer_out.shape[-1],
                            moshi_transformer_out.shape[-1],
                            device=ttt_transformer_out.device,
                            dtype=ttt_transformer_out.dtype
                        )
                        nn.init.normal_(self.ttt_output_projection.weight, std=0.02)
                        nn.init.zeros_(self.ttt_output_projection.bias)
                    
                    # Project TTT output to match dimensions
                    ttt_transformer_out = self.ttt_output_projection(ttt_transformer_out)
                    logger.info(f"Projected TTT output shape: {ttt_transformer_out.shape}")
                elif hasattr(self, 'ttt_projection'):
                    # Use existing ttt_projection if available
                    ttt_transformer_out = self.ttt_projection(ttt_transformer_out)
                    logger.info(f"Applied existing ttt_projection, output shape: {ttt_transformer_out.shape}")
                
                # Verify dimensions match before combining
                assert ttt_transformer_out.shape == moshi_transformer_out.shape, \
                    f"Dimension mismatch: TTT output {ttt_transformer_out.shape}, Moshi output {moshi_transformer_out.shape}"
                
                # Get model's expected dtype
                model_dtype = self.dtype
                
                # Ensure both tensors have the same dtype before weighted combination
                if moshi_transformer_out.dtype != model_dtype:
                    logger.info(f"Converting moshi_transformer_out from {moshi_transformer_out.dtype} to {model_dtype}")
                    moshi_transformer_out = moshi_transformer_out.to(model_dtype)
                    
                if ttt_transformer_out.dtype != model_dtype:
                    logger.info(f"Converting ttt_transformer_out from {ttt_transformer_out.dtype} to {model_dtype}")
                    ttt_transformer_out = ttt_transformer_out.to(model_dtype)
                
                # Use weighted combination of outputs
                transformer_out = self.ttt_integration_weight * moshi_transformer_out + (1 - self.ttt_integration_weight) * ttt_transformer_out
                logger.info(f"Combined output shape: {transformer_out.shape}, "
                        f"integration weight: {self.ttt_integration_weight}")
            except Exception as e:
                # Log error and fall back to using just the original transformer output
                logger.error(f"TTT processing failed with error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("Falling back to using only the original transformer output")
                transformer_out = moshi_transformer_out
        else:
            # If TTT is disabled, just use the original transformer output
            transformer_out = moshi_transformer_out
        
        # Apply normalization if available
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
            # RMSNorm might convert to float32 internally for precision; ensure we convert back if needed
            if transformer_out.dtype != self.dtype:
                logger.debug(f"Converting transformer_out back to {self.dtype} after out_norm")
                transformer_out = transformer_out.to(self.dtype)
        
        assert isinstance(transformer_out, torch.Tensor)
        
        # Safely determine the dtype of text_linear, which could be a LoRALinear or nn.Linear
        if hasattr(self.text_linear, "__class__") and self.text_linear.__class__.__name__ == "LoRALinear":
            # If it's a LoRALinear, get dtype from one of its components
            if hasattr(self.text_linear, 'lora_A') and hasattr(self.text_linear.lora_A, 'weight'):
                text_linear_dtype = self.text_linear.lora_A.weight.dtype
            elif hasattr(self.text_linear, 'frozen_W') and hasattr(self.text_linear.frozen_W, 'weight'):
                text_linear_dtype = self.text_linear.frozen_W.weight.dtype
            else:
                # Fallback - get dtype from the first parameter
                try:
                    text_linear_dtype = next(self.text_linear.parameters()).dtype
                except StopIteration:
                    logger.warning("Could not determine dtype for LoRALinear text_linear. Using transformer_out dtype.")
                    text_linear_dtype = transformer_out.dtype
        elif hasattr(self.text_linear, 'weight'):
            # Standard nn.Linear
            text_linear_dtype = self.text_linear.weight.dtype
        else:
            # Ultimate fallback
            logger.warning("Could not access weight attribute on text_linear. Using transformer_out dtype.")
            text_linear_dtype = transformer_out.dtype
            
        # Ensure dtypes match before linear transformation
        if transformer_out.dtype != text_linear_dtype:
            logger.debug(f"Converting transformer_out from {transformer_out.dtype} to {text_linear_dtype} before text_linear")
            transformer_out = transformer_out.to(text_linear_dtype)
            
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        
        logger.info(f"LMModel.forward_text: Returning transformer_out shape {transformer_out.shape} and text_logits shape {text_logits.shape}")
        
        return transformer_out, text_logits


    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        logger.info("LMModel.forward_depformer_training: Starting forward_depformer_training")
        B, K, T = sequence.shape
        Ka = self.dep_q
        assert (
            K == self.num_codebooks
        ), f"Codebooks for Depformer training should be passed all at once, got {K,}."
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        
        logger.info(f"LMModel.forward_depformer_training: Returning logits with shape {logits.shape}")
        
        return logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        # Note: transformer_out is already the processed output from TTTModel + original transformer
        # No changes needed to this method's logic
        
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            in_index = depformer_cb_index
            if self.depformer_weights_per_step_schedule is not None:
                in_index = self.depformer_weights_per_step_schedule[in_index]
            depformer_input = self.depformer_in[in_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        assert last_token_input is not None
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def _init_weights(self):
        """Initialization of the transformer module weights.
        Mostly truncated gaussian, with `std = 1 / sqrt(dim_in)`.
        Embeddings are also initialized with `1 / sqrt(dim)` rather than `1`.
        Some layers are not going to be properly initialized:
            - in_proj in MHA.
            - depth transformer layers.
        This is to match how our models were trained so far.
        """

        for emb_layer in self.emb:
            _init_layer(emb_layer)
        for emb_layer in self.depformer_emb:
            _init_layer(emb_layer)
        _init_layer(self.text_emb)
        _init_layer(self.depformer_text_emb)
        _init_layer(self.text_linear)

        for tr_layer in self.transformer.layers:
            tr_layer.apply(_init_layer)

        for linear in self.linears:
            _init_layer(linear)

        # Initialize TTTModel components
        if hasattr(self, 'user_ttt_model'):
            self.user_ttt_model.apply(_init_layer)
        if hasattr(self, 'ttt_projection'):
            _init_layer(self.ttt_projection)

    def set_streaming_detached(self, detached: bool) -> None:
        """Set whether this module is detached from parent streaming state changes.
        
        When detached is set to True, this module will not follow parent's streaming state.
        We also use this opportunity to clear the managed TTT cache when detaching.
        """
        super().set_streaming_detached(detached)
        # TTT should follow the main transformer's streaming state
        self.depformer.set_streaming_detached(detached) 
        # Clear the managed TTT cache when detaching or attaching
        if hasattr(self, '_managed_ttt_cache'):
            logger.info("LMModel: Clearing managed TTTCache during streaming detachment change.")
            self._managed_ttt_cache = None
    
    def reset_streaming(self, reset_mask: torch.Tensor) -> None:
        """Reset the streaming state for sequences specified by reset_mask.
        
        This is called when some sequences in a batch need to be reset, for example
        when a conversation ends and a new one begins in the same batch slot.
        
        Args:
            reset_mask: A boolean tensor of shape [batch_size] where True indicates
                       that the corresponding sequence should be reset.
        """
        super().reset_streaming(reset_mask)
        
        # Reset the managed TTT cache for specified sequences
        if hasattr(self, '_managed_ttt_cache') and self._managed_ttt_cache is not None:
            logger.info(f"LMModel: Resetting managed TTTCache for {reset_mask.sum().item()} sequences.")
            
            # Get the batch size from the existing cache
            batch_size = self._managed_ttt_cache.ttt_params_dict["W1_states"][0].shape[0]
            
            # Store the current cache
            old_cache = self._managed_ttt_cache
            
            # Create new cache with the same batch size
            self._managed_ttt_cache = TTTCache(self.user_ttt_model, batch_size)
            
            # Only copy states for sequences that weren't reset
            not_reset = ~reset_mask
            if not_reset.any():
                for layer_idx in range(self.user_ttt_model.config.num_hidden_layers):
                    for name in self._managed_ttt_cache.ttt_param_names:
                        for key in [f"{name}_states", f"{name}_grad"]:
                            self._managed_ttt_cache.ttt_params_dict[key][layer_idx][not_reset] = old_cache.ttt_params_dict[key][layer_idx][not_reset]
                
                # For sequences that weren't reset, keep their original sequence length offset
                # Note: TTTCache has a global seqlen_offset, so we can't have different offsets per sequence
                # We maintain the old offset as a compromise
                self._managed_ttt_cache.seqlen_offset = old_cache.seqlen_offset
            else:
                # If all sequences were reset, start from offset 0
                self._managed_ttt_cache.seqlen_offset = 0

    def _init_streaming_state(self, batch_size: int) -> LMModelState:
        """Initialize the streaming state for LMModel.
        This is called when entering streaming mode via LMGen.streaming() context manager.
        We use this to create and initialize our managed TTT cache.
        """
        logger.debug(f"LMModel._init_streaming_state called with batch_size: {batch_size}")
        
        # Initialize the managed TTT cache if TTT model is available and configured to use cache
        if hasattr(self, 'user_ttt_model') and self.user_ttt_model and self.user_ttt_model.config.use_cache:
            logger.info(f"LMModel: Initializing managed TTTCache with batch_size: {batch_size}")
            # TTTCache constructor is TTTCache(model: TTTModel, batch_size: int)
            self._managed_ttt_cache = TTTCache(self.user_ttt_model, batch_size)
        else:
            logger.debug("LMModel: Not initializing TTTCache (TTT model not available or use_cache disabled)")
            self._managed_ttt_cache = None
            
        # Return a proper LMModelState object
        return LMModelState(batch_size, self.device)
        
    def _set_streaming_state(self, state: tp.Optional[LMModelState]):
        """Set or clear the streaming state for LMModel.
        This is called when the streaming context might be resetting or ending.
        We use this to clear our managed TTT cache when appropriate.
        """
        logger.debug(f"LMModel._set_streaming_state called with state: {type(state)}")
        
        # If state is None, it typically indicates end of streaming context or reset
        if state is None:
            logger.info("LMModel: Clearing managed TTTCache as streaming context ends.")
            self._managed_ttt_cache = None


@dataclass
class _LMGenState(State):
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    offsets: torch.Tensor
    offset_cpu: int = 0
    condition_sum: torch.Tensor | None = None
    condition_cross: torch.Tensor | None = None
    exit_stack: ExitStack = field(default_factory=ExitStack)
    reset_callback: tp.Callable[[torch.Tensor], None] | None = None

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        self.offsets[:] = torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)
        self.offset_cpu = 0
        if self.reset_callback is not None:
            self.reset_callback(reset_mask)

    def __enter__(self):
        self.exit_stack.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.__exit__(exc_type, exc_value, traceback)


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        cfg_coef: float = 1.,
        check: bool = False,
        condition_tensors: ConditionTensors | None = None,
        on_text_hook: tp.Optional[tp.Callable[[torch.Tensor], None]] = None,
        on_audio_hook: tp.Optional[tp.Callable[[torch.Tensor], None]] = None,
        support_out_of_sync: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.lm_model.set_streaming_detached(True)
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.cfg_coef = cfg_coef
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.condition_tensors = condition_tensors
        self.on_text_hook = on_text_hook
        self.on_audio_hook = on_audio_hook
        self.support_out_of_sync = support_out_of_sync
        if self.cfg_coef != 1.:
            assert self.lm_model.fuser is not None, "Model has no fuser, cannot do CFG."
            assert self.condition_tensors, "Missing condition tensors for CFG."

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )
        offsets = torch.zeros(batch_size, device=lm_model.device, dtype=torch.long)

        if self.lm_model.fuser is None:
            assert not self.condition_tensors
            condition_sum = None
            condition_cross = None
        else:
            assert self.condition_tensors is not None
            condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)
            condition_cross = self.lm_model.fuser.get_cross(self.condition_tensors)
            if condition_sum is not None:
                condition_sum = condition_sum.to(self.lm_model.dtype)
            if condition_cross is not None:
                condition_cross = condition_cross.to(self.lm_model.dtype)

        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=True)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=True)

        state = _LMGenState(
            batch_size, lm_model.device, cache, initial, graphed_main, graphed_depth,
            offsets, condition_sum=condition_sum, condition_cross=condition_cross)

        if self.cfg_coef != 1.:
            batch_size *= 2
            if state.condition_sum is not None:
                assert state.condition_sum.shape[0] == batch_size, "cfg requires 2x more conditions."
            if state.condition_cross is not None:
                assert state.condition_cross.shape[0] == batch_size, "cfg requires 2x more conditions."
        state.exit_stack.enter_context(self.lm_model.streaming(batch_size))

        def _reset_callback(reset_mask: torch.Tensor) -> None:
            if self.cfg_coef != 1.:
                reset_mask = reset_mask.repeat(2)
            self.lm_model.reset_streaming(reset_mask)
        state.reset_callback = _reset_callback
        return state

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor,
             depformer_replace_tokens: torch.Tensor | None = None) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
        assert S == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki >= needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        if Ki > needed_tokens:
            input_tokens = input_tokens[:, :needed_tokens, :]

        CT = state.cache.shape[2]

        delays = self.delays_cuda[lm_model.dep_q + 1:]
        write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
        state.cache[:, lm_model.dep_q + 1:].scatter_(-1, write_positions, input_tokens)

        is_init = state.offsets[:, None, None] <= self.delays_cuda[:, None]
        is_init |= ~state.exec_mask[:, None, None]  # we also give init tokens if not executing to avoid crashing.
        positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
        input_ = state.cache.gather(dim=2, index=positions)
        input_ = torch.where(is_init, state.initial, input_)

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offsets,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        if self.cfg_coef != 1.:
            input_ = input_.repeat(2, 1, 1)
        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum, state.condition_cross)
        if self.cfg_coef != 1.:
            logits, logits_null = text_logits.chunk(2)
            text_logits = logits_null + (logits - logits_null) * self.cfg_coef
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]
        if self.on_text_hook is not None:
            self.on_text_hook(text_token)
        if depformer_replace_tokens is None:
            audio_tokens = state.graphed_depth(text_token, transformer_out)
            if self.on_audio_hook is not None:
                self.on_audio_hook(audio_tokens)
        else:
            assert depformer_replace_tokens.dim() == 3
            audio_tokens = depformer_replace_tokens.squeeze(-1)

        state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
        state.offset_cpu += 1
        positions = state.offsets % CT
        state.cache[:, :1].scatter_(-1, positions[:, None, None], text_token[:, None, None])
        audio_tokens = audio_tokens[:, :, None]
        state.cache[:, 1: lm_model.dep_q + 1, :].scatter_(
            -1, positions[:, None, None].expand_as(audio_tokens), audio_tokens)

        if not self.support_out_of_sync and state.offset_cpu <= self.max_delay:
            # When using out of sync exec, should not rely on this being None.
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (state.offsets[:, None, None] - self.max_delay + gen_delays_cuda[:, None]) % CT
        out = state.cache.gather(dim=2, index=index)
        mask = (state.offsets <= self.max_delay) | ~state.exec_mask
        out[mask, :, :] = lm_model.ungenerated_token_id
        return out

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, = text_token.shape
        B_cfg = B
        if self.cfg_coef != 1.:
            B_cfg = 2 * B
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B_cfg):
            assert lm_model.depformer.is_streaming
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                if self.cfg_coef != 1.:
                    input_ = input_.repeat(2, 1, 1)
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.cfg_coef != 1.:
                    logits, logits_null = logits.chunk(2)
                    logits = logits_null + (logits - logits_null) * self.cfg_coef
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out
