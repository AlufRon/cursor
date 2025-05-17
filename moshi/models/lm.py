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
from .ttt import TTTModel, TTTConfig, TTTCache
import math


logger = logging.getLogger(__name__)


def scatter_with_mask_(tensor: torch.Tensor, dim: int,
                       index: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> None:
    """Scatter but skipping the updates that are masked."""
    old_value = tensor.gather(dim, index)
    value = torch.where(mask, value, old_value)
    tensor.scatter_(dim, index, value)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


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
        use_ttt (bool): If True, use TTT model in parallel with temporal transformer.
        ttt_config_overrides (dict, optional): Dictionary of parameters to override default TTT configuration.
        ttt_integration_mode (str): How to combine TTT and transformer outputs. Options: 'concat', 'weighted_sum'.
        ttt_integration_weight (float): Weight for TTT output when using 'weighted_sum' integration mode.
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
        use_ttt: bool = False,
        ttt_config_overrides: tp.Optional[dict] = None,
        ttt_integration_mode: str = 'concat',
        ttt_integration_weight: float = 0.5,
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
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        
        # Store TTT Control Flags
        self.use_ttt = use_ttt
        self.ttt_integration_mode = ttt_integration_mode
        self.ttt_integration_weight = ttt_integration_weight # Only used if mode is 'weighted_sum'
        
        # Original kwargs_dep setup - needed for depformer later
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
        
        self.user_ttt_model: tp.Optional[TTTModel] = None
        self.user_ttt_config: tp.Optional[TTTConfig] = None
        self.ttt_projection: tp.Optional[nn.Module] = None
        self._managed_ttt_cache: tp.Optional[TTTCache] = None # For streaming state
        
        if self.use_ttt:
            logger.info(f"TTT Integration Enabled. Mode: {self.ttt_integration_mode}")
            
            # --- Define TTTConfig ---
            # These are example defaults; make them configurable via ttt_config_overrides
            # or pass directly to LMModel __init__
            default_ttt_params = {
                "hidden_size": self.dim,  # Match Moshi's internal dimension
                "intermediate_size": int(hidden_scale * self.dim), # Example based on Moshi's hidden_scale
                "num_hidden_layers": kwargs.get('ttt_num_layers', 6), # Allow override via kwargs
                "num_attention_heads": num_heads, # Match Moshi's num_heads
                "max_position_embeddings": self.context if self.context is not None else 2048,
                "vocab_size": self.text_card + 1,  # Placeholder, as TTTModel will receive embeddings
                "ttt_layer_type": kwargs.get('ttt_layer_type', 'linear'),
                "ttt_base_lr": kwargs.get('ttt_base_lr', 1.0),
                "mini_batch_size": kwargs.get('ttt_mini_batch_size', 16), # TTT's internal mini-batch
                "use_cache": True,  # Necessary for streaming state management via TTTCache
                "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, # Required by TTTConfig
                "pre_conv": kwargs.get('ttt_pre_conv', False),
                "conv_kernel": kwargs.get('ttt_conv_kernel', 4),
                "use_gate": kwargs.get('ttt_use_gate', False),
                "share_qk": kwargs.get('ttt_share_qk', False),
                "scan_checkpoint_group_size": kwargs.get('ttt_scan_checkpoint_group_size', 0),
            }
            if ttt_config_overrides: # Allow a dictionary to override these defaults
                default_ttt_params.update(ttt_config_overrides)

            self.user_ttt_config = TTTConfig(**default_ttt_params)
            self.user_ttt_model = TTTModel(self.user_ttt_config)

            # --- TTT Output Projection (Optional but recommended) ---
            # This layer projects TTT output to self.dim if different, or can be a learned mapping
            if self.user_ttt_config.hidden_size != self.dim:
                self.ttt_projection = nn.Linear(self.user_ttt_config.hidden_size, self.dim, bias=False)
            else:
                # Even if dims match, a projection might be useful for learning how to best integrate.
                # For simplicity, start with Identity if dims match.
                self.ttt_projection = nn.Identity()
            logger.info(f"TTT model initialized (hidden_size: {self.user_ttt_config.hidden_size}). Projection type: {type(self.ttt_projection)}")
        else:
            logger.info("TTT integration is disabled.")

        # Calculate the effective input dimension for depformer_in
        if self.use_ttt:
            if self.ttt_integration_mode == 'concat':
                # Assuming ttt_projection outputs self.dim
                effective_input_dim_to_depformer = self.dim + self.dim
            elif self.ttt_integration_mode == 'weighted_sum':
                # For weighted sum, both original and TTT (projected) outputs must be self.dim
                effective_input_dim_to_depformer = self.dim
            else:
                raise ValueError(f"Unknown ttt_integration_mode: {self.ttt_integration_mode}")
        else: # No TTT
            effective_input_dim_to_depformer = self.dim

        logger.info(f"Effective input dimension to depformer_in layers: {effective_input_dim_to_depformer}")

        # Now, re-initialize self.depformer_in using this effective dimension
        # The structure (multi_linear or single) remains the same.
        self.depformer_multi_linear = depformer_multi_linear # Make sure this is set

        # Determine the number of projection layers needed for depformer_in
        if self.depformer_multi_linear:
            num_dep_in_projections = self.dep_q # Original uses self.dep_q
            if self.depformer_weights_per_step_schedule is not None:
                num_dep_in_projections = max(self.depformer_weights_per_step_schedule) + 1
        else:
            num_dep_in_projections = 1

        self.depformer_in = nn.ModuleList()
        for _ in range(num_dep_in_projections):
            self.depformer_in.append(
                nn.Linear(effective_input_dim_to_depformer, depformer_dim, bias=False)
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
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)

    def forward_text(
        self,
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
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
        
        # Run the original transformer
        transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        
        # Process TTT model in parallel if enabled
        if self.use_ttt:
            logger.info(f"Running TTT model in forward_text (integration mode: {self.ttt_integration_mode})")
            # Generate text logits from transformer alone (for output)
            text_logits = self.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            
            # Generate position IDs for TTT (required by TTT model)
            position_ids = torch.arange(S, device=input_.device).unsqueeze(0).expand(B, -1)
            
            # Initialize TTT cache if in streaming mode
            # Note: For non-streaming usage (training), cache_params is None
            cache_params = None
            if hasattr(self, '_managed_ttt_cache') and self._managed_ttt_cache is not None:
                cache_params = self._managed_ttt_cache
            
            # Process through TTT model
            # Input to TTT is the same embedding that went to the transformer
            # This could be modified to pass a different representation if needed
            logger.info(f"TTT forward pass with seq_len={S}, batch_size={B}")
            
            # Create attention mask since we're not providing input_ids
            attention_mask = torch.ones((B, S), dtype=torch.long, device=input_.device)
            
            ttt_out = self.user_ttt_model(
                inputs_embeds=input_, 
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_params=cache_params,
                return_dict=True
            )
            
            # Get TTT's final hidden states
            ttt_hidden = ttt_out.last_hidden_state
            
            # Update cache if provided
            if cache_params is not None:
                self._managed_ttt_cache = ttt_out.cache_params
                logger.info("TTT cache updated in forward_text")
            
            # Project TTT output to match transformer dimension if needed
            ttt_projected = self.ttt_projection(ttt_hidden)
            
            # Combine TTT and transformer outputs based on integration mode
            if self.ttt_integration_mode == 'concat':
                # Concatenate along the feature dimension
                combined_out = torch.cat([transformer_out, ttt_projected], dim=-1)
                logger.info(f"TTT output concatenated with transformer output -> shape: {combined_out.shape}")
            elif self.ttt_integration_mode == 'weighted_sum':
                # Weighted sum of the two outputs
                transformer_weight = 1.0 - self.ttt_integration_weight
                combined_out = (transformer_weight * transformer_out) + (self.ttt_integration_weight * ttt_projected)
                logger.info(f"TTT output weighted sum (w={self.ttt_integration_weight}) with transformer output")
            else:
                # Should never happen as we validate in __init__, but just in case
                raise ValueError(f"Unsupported ttt_integration_mode: {self.ttt_integration_mode}")
            
            # Return the combined output and the text logits
            return combined_out, text_logits
        else:
            # Original behavior when TTT is disabled
            text_logits = self.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
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
        return logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
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
        
        # Check if we should process TTT model for streaming
        if self.use_ttt and hasattr(self, '_managed_ttt_cache'):
            logger.info(f"Running TTT model in streaming (forward_depformer) for cb_index={depformer_cb_index}")
            # Generate position IDs for TTT
            position_ids = torch.zeros((B, 1), device=transformer_out.device, dtype=torch.long)
            
            # Get input embedding from the stream cache if possible, otherwise create it
            # This is simplified and might need to be adjusted based on how streaming works
            # A proper implementation would extract input embeddings from the transformer cache
            if hasattr(self, '_last_input_embedding'):
                input_embeds = self._last_input_embedding
            else:
                # Fallback - not ideal but provides a mechanism if embedding not cached
                # In real streaming, we should have a better mechanism to retrieve the input
                logger.warning("Using fallback input embedding generation for TTT - implement a proper cache")
                if depformer_cb_index == 0:
                    input_embeds = self.text_emb(sequence[:, 0])
                else:
                    audio_idx = depformer_cb_index - 1 + self.audio_offset
                    input_embeds = self.emb[audio_idx - 1](sequence[:, 0])
            
            # Process through TTT model with the managed cache
            logger.info(f"TTT streaming forward pass for batch_size={B}")
            
            # Create attention mask since we're not providing input_ids
            attention_mask = torch.ones((B, 1), dtype=torch.long, device=input_embeds.device)
            
            ttt_out = self.user_ttt_model(
                inputs_embeds=input_embeds.unsqueeze(1),  # Add sequence dimension
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_params=self._managed_ttt_cache,
                return_dict=True
            )
            
            # Update TTT cache
            self._managed_ttt_cache = ttt_out.cache_params
            logger.info("TTT cache updated in streaming")
            
            # Project TTT output
            ttt_projected = self.ttt_projection(ttt_out.last_hidden_state)
            
            # Combine with transformer output based on integration mode
            if self.ttt_integration_mode == 'concat':
                # Concatenate along feature dimension
                combined_out = torch.cat([transformer_out, ttt_projected], dim=-1)
                logger.info(f"TTT streaming: outputs concatenated -> shape: {combined_out.shape}")
            elif self.ttt_integration_mode == 'weighted_sum':
                # Weighted sum
                transformer_weight = 1.0 - self.ttt_integration_weight
                combined_out = (transformer_weight * transformer_out) + (self.ttt_integration_weight * ttt_projected)
                logger.info(f"TTT streaming: weighted sum (w={self.ttt_integration_weight})")
            else:
                raise ValueError(f"Unsupported ttt_integration_mode: {self.ttt_integration_mode}")
                
            # Use the combined output for depformer processing
            depformer_input = combined_out
        else:
            # Original path when TTT is disabled
            depformer_input = transformer_out
        
        # Continue with the original logic using the (potentially combined) depformer_input
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
            
        if self.use_ttt:
            if self.user_ttt_model is not None:
                logger.info("Applying _init_weights to self.user_ttt_model")
                # TTTModel has its own _init_weights method.
                # It's generally better to call the model's own initializer if it exists and is comprehensive.
                if hasattr(self.user_ttt_model, '_init_weights') and callable(self.user_ttt_model._init_weights):
                    self.user_ttt_model.apply(self.user_ttt_model._init_weights)
                else:
                    # Fallback: Apply Moshi's _init_layer to all submodules of TTTModel.
                    # This might not be ideal if TTTModel contains layers not handled by _init_layer (e.g., RMSNorm).
                    logger.warning("TTTModel does not have a callable _init_weights method. Applying generic _init_layer. Review carefully.")
                    self.user_ttt_model.apply(_init_layer) # Ensure _init_layer is defined in LMModel or accessible

            if self.ttt_projection is not None and isinstance(self.ttt_projection, nn.Linear):
                logger.info("Initializing self.ttt_projection")
                _init_layer(self.ttt_projection) # Use the existing _init_layer for Linear


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
    set_exec_mask_callback: tp.Callable[[torch.Tensor], None] | None = None
    # Tracking the last TTT cache reset to avoid unnecessary resets
    last_ttt_cache_reset: bool = False

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        self.offsets[:] = torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)
        self.offset_cpu = 0
        
        # Reset TTT cache if applicable
        if reset_mask.any() and not self.last_ttt_cache_reset:
            # In a streaming generation context, we need to access the parent LMGen instance
            # to get the lm_model reference. We store this temporarily during reset.
            # This approach avoids circular imports.
            from ..modules.streaming import get_active_module
            active_module = get_active_module()
            if active_module is not None and hasattr(active_module, 'lm_model'):
                lm_model = active_module.lm_model
                if hasattr(lm_model, 'use_ttt') and lm_model.use_ttt:
                    if hasattr(lm_model, '_managed_ttt_cache') and lm_model._managed_ttt_cache is not None:
                        logger.info("Resetting TTT cache during state reset")
                        logger.info(f"TTT model active with {lm_model.user_ttt_config.num_hidden_layers} layers")
                        # Create a new TTT cache for the model, effectively resetting it
                        # Note: In a production implementation, we might want a more fine-grained reset
                        # mechanism that only affects specific batch items based on reset_mask
                        effective_batch = self.batch_size
                        if hasattr(active_module, 'cfg_coef') and getattr(active_module, 'cfg_coef', 1.0) != 1.0:
                            effective_batch *= 2
                        lm_model._managed_ttt_cache = TTTCache(
                            model=lm_model.user_ttt_model,
                            batch_size=effective_batch
                        )
                        self.last_ttt_cache_reset = True
        elif not reset_mask.any():
            # Reset the tracking flag when no reset is happening
            self.last_ttt_cache_reset = False
        
        if self.reset_callback is not None:
            self.reset_callback(reset_mask)

    def set_exec_mask(self, exec_mask: torch.Tensor):
        super().set_exec_mask(exec_mask)
        if self.set_exec_mask_callback is not None:
            self.set_exec_mask_callback(exec_mask)

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
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        # Initialize TTT cache if TTT is enabled
        if lm_model.use_ttt and lm_model.user_ttt_model is not None:
            logger.info("Initializing TTT cache for streaming")
            logger.info(f"TTT model using {lm_model.ttt_integration_mode} integration with weight={lm_model.ttt_integration_weight}")
            # Create a TTT cache for streaming
            if hasattr(lm_model, '_managed_ttt_cache'):
                logger.info("Reusing existing TTT cache")
            else:
                # Note: TTTCache takes a model and batch size
                effective_batch = batch_size
                if self.cfg_coef != 1.:
                    effective_batch *= 2  # Double for CFG
                
                lm_model._managed_ttt_cache = TTTCache(
                    model=lm_model.user_ttt_model, 
                    batch_size=effective_batch
                )
                logger.info(f"Created new TTT cache with batch size {effective_batch}")
                
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

        def _set_exec_mask_callback(exec_mask: torch.Tensor) -> None:
            if self.cfg_coef != 1.:
                exec_mask = exec_mask.repeat(2)
            self.lm_model.set_exec_mask(exec_mask)

        state.reset_callback = _reset_callback
        state.set_exec_mask_callback = _set_exec_mask_callback
        return state

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor,
             depformer_replace_tokens: torch.Tensor | None = None) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        
        # Log every 10 steps to show TTT status
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
            if self._step_counter % 10 == 0:
                if self.lm_model.use_ttt:
                    logger.info(f"Generation step {self._step_counter} with TTT enabled")
        else:
            self._step_counter = 0
            if self.lm_model.use_ttt:
                logger.info("Starting generation with TTT enabled")
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
        scatter_with_mask_(state.cache[:, lm_model.dep_q + 1:], -1, write_positions, input_tokens,
                           state.exec_mask[:, None, None])

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
        positions = (state.offsets % CT)[:, None, None]
        scatter_with_mask_(state.cache[:, :1], -1, positions,
                           text_token[:, None, None], state.exec_mask[:, None, None])
        audio_tokens = audio_tokens[:, :, None]
        scatter_with_mask_(state.cache[:, 1: lm_model.dep_q + 1, :], -1,
                           positions.expand_as(audio_tokens),
                           audio_tokens,
                           state.exec_mask[:, None, None])

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
