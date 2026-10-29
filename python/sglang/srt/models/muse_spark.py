"""Muse Spark text model for SGLang, backported from upstream main.

A torch-native port of upstream ``models/muse_spark.py`` that only uses
operators present in this release: the attention runs on the configured
backend through ``RadixAttention`` and the routed experts run on the Triton
fused MoE kernel. The vision and audio towers are not loaded; the model
serves text.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.srt.configs.muse_spark import (
    MuseSparkTextConfig,
    should_skip_weight_loading,
)
from sglang.srt.distributed import divide, tensor_model_parallel_all_reduce
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    resolve_linear_parallel_group,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

_EXPERT_SHARD_IDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0)
            y = x.to(tl.float32) * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    slot_expert_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``sum_k score_k * rms_norm(weight * x_k)`` over ``[T, top_k, H]`` outputs."""
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape
    if tokens < _MIN_TOKENS_FOR_FUSED:
        y = x.float() * norm_weight.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        scores = topk_scores.float()
        if slot_expert_ids is not None:
            scores = torch.where(slot_expert_ids >= 0, scores, 0.0)
        return (y * scores.unsqueeze(-1)).sum(dim=1).to(x.dtype)
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight.contiguous(),
        topk_scores.contiguous(),
        out,
        slot_expert_ids if slot_expert_ids is not None else x,
        TOPK=topk,
        H=hidden,
        eps=eps,
        HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


class MuseSpark1xRMSNorm(nn.Module):
    """RMSNorm with an optional zero-centered gain (weight + gain_center)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        zero_centered_gamma: bool = False,
        gain_center: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.gain_center = gain_center
        self.weight = nn.Parameter(torch.empty(hidden_size)) if has_weight else None
        object.__setattr__(self, "_frozen_weight", None)

    def effective_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None or not self.zero_centered_gamma:
            return self.weight
        if self._frozen_weight is not None:
            return self._frozen_weight
        return self.weight + self.gain_center

    @torch.no_grad()
    def precompute_effective_weight(self) -> None:
        if self.weight is not None and self.zero_centered_gamma:
            object.__setattr__(
                self, "_frozen_weight", (self.weight + self.gain_center).detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        weight = self.effective_weight()
        if weight is not None:
            weight = weight.to(x.dtype)
        return F.rms_norm(x, (self.hidden_size,), weight, self.eps)


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    """Scales the input by the gain before a weightless RMSNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        dtype = x.dtype
        weight = self.effective_weight()
        if weight is not None:
            x = weight * x
        return F.rms_norm(x, (self.hidden_size,), None, self.eps).to(dtype)


class MuseSpark1xNormPreservingResidualConnection(nn.Module):
    """Gated residual: ``alpha * stream + beta * branch``."""

    def __init__(self, dim: int, temperature: float = 0.3, eps: float = 1e-3):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim))
        self.register_buffer("_alpha", None, persistent=False)
        self.register_buffer("_beta", None, persistent=False)

    def _coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.gate.view(1, -1).float() / self.temperature
        beta = torch.sigmoid(w)
        alpha_sq = torch.sigmoid(-w) * (1.0 + beta)
        return torch.sqrt(torch.clamp(alpha_sq, min=self.eps)), beta

    @torch.no_grad()
    def precompute_coefficients(self) -> None:
        self._alpha, self._beta = self._coefficients()

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if stream.numel() == 0:
            return stream
        if self._alpha is None:
            alpha, beta = self._coefficients()
        else:
            alpha, beta = self._alpha, self._beta
        return (alpha * stream + beta * branch.to(stream.dtype)).to(stream.dtype)


class QKVOGateParallelLinear(ColumnParallelLinear):
    """Tensor-parallel Q/K/V projection with one additional query-sized gate."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        prefix: str = "",
        parallel_group="attn_tp",
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        _, tp_size = resolve_linear_parallel_group(parallel_group)
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * head_size
        self.kv_proj_shard_size = self.num_kv_heads * head_size
        self.g_proj_shard_size = self.q_proj_shard_size
        local = (
            self.q_proj_shard_size
            + 2 * self.kv_proj_shard_size
            + self.g_proj_shard_size
        )
        super().__init__(
            input_size=hidden_size,
            output_size=local * tp_size,
            bias=bias,
            gather_output=False,
            quant_config=None,
            prefix=prefix,
            parallel_group=parallel_group,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        if loaded_shard_id not in ("q", "k", "v", "g"):
            raise ValueError(f"Unsupported QKV gate shard: {loaded_shard_id}")
        q, kv = self.q_proj_shard_size, self.kv_proj_shard_size
        sizes = {"q": q, "k": kv, "v": kv, "g": q}
        offsets = {"q": 0, "k": q, "v": q + kv, "g": q + 2 * kv}
        output_dim = getattr(param, "output_dim", 0)
        tp_rank, _ = get_group_rank_size(self.tp_group)
        shard_size = sizes[loaded_shard_id]
        param_data = param.data.narrow(output_dim, offsets[loaded_shard_id], shard_size)
        if loaded_shard_id in ("q", "g"):
            start = tp_rank * shard_size
        else:
            start = (tp_rank // self.num_kv_head_replicas) * shard_size
        param_data.copy_(loaded_weight.narrow(output_dim, start, shard_size))


class MuseSpark1xAttention(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.use_rope = config.layer_rope_theta[layer_idx] > 0
        sliding = config.layer_types[layer_idx] == "sliding_attention"

        tp_size = get_parallel().attn_tp_size
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # RoPE is a rotation, so the post-QK-norm scalar folds into the softmax scale.
        self.scaling = self.head_dim**-0.5 * config.qk_scale_factor
        self.eps = config.rms_norm_eps

        self.qkv_proj = QKVOGateParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
            parallel_group="attn_tp",
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=getattr(
                    config, "context_len", config.max_position_embeddings
                ),
                base=config.layer_rope_theta[layer_idx],
                is_neox_style=False,
                dtype=torch.float32,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=config.sliding_window - 1 if sliding else -1,
        )

    def _head_norm(self, x: torch.Tensor) -> torch.Tensor:
        d = self.head_dim
        return F.rms_norm(x.reshape(-1, d), (d,), None, self.eps).reshape(x.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = qkv.split(
            [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
        )
        q = self._head_norm(q)
        k = self._head_norm(k)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v.contiguous(), forward_batch)
        output = gate.sigmoid() * self._head_norm(output)
        output, _ = self.o_proj(output)
        return output


class MuseSpark1xMLP(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MuseSpark1xRouter(nn.Module):
    """FP32 router in HF linear layout ``[experts, hidden]`` plus selection bias."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight)


class MuseSpark1xExperts(nn.Module):
    """Local BF16 expert stacks, run by the Triton fused MoE kernel."""

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        local_expert_offset: int,
        expert_hidden_size: int,
        intermediate_size: int,
        top_k: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_local_experts, 2 * intermediate_size, expert_hidden_size),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_local_experts, expert_hidden_size, intermediate_size),
            requires_grad=False,
        )
        for param in (self.gate_up_proj, self.down_proj):
            param.weight_loader = self.weight_loader
        self.runner_config = MoeRunnerConfig(
            num_experts=num_local_experts,
            num_local_experts=num_local_experts,
            hidden_size=expert_hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            no_combine=True,
            gate_up_interleaved=False,
        )

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        local_id = expert_id - self.local_expert_offset
        if not 0 <= local_id < self.num_local_experts:
            return
        target = param.data[local_id]
        if shard_id in ("w1", "w3"):
            offset = 0 if shard_id == "w1" else self.intermediate_size
            target = target.narrow(0, offset, self.intermediate_size)
        if target.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {weight_name}: "
                f"parameter={tuple(target.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: MuseSpark1xRMSNorm,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts

        offset = self.local_expert_offset
        is_local = (topk_ids >= offset) & (topk_ids < offset + self.num_local_experts)
        local_ids = torch.where(is_local, topk_ids - offset, 0).to(torch.int32)
        # no_combine returns each slot's expert output [T, top_k, H] unweighted.
        expert_output = fused_experts(
            hidden_states=hidden_states.contiguous(),
            w1=self.gate_up_proj,
            w2=self.down_proj,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones_like(topk_scores, dtype=torch.float32),
                topk_ids=local_ids.contiguous(),
                router_logits=None,
            ),
            moe_runner_config=self.runner_config,
        )
        # Muse norms each expert output before the weighted top-k sum.
        return fused_post_expert_norm_weighted_sum(
            expert_output,
            topk_scores.float(),
            norm_weight=post_expert_norm.effective_weight().to(torch.bfloat16),
            eps=float(post_expert_norm.eps),
            slot_expert_ids=(
                None
                if self.num_local_experts == self.num_experts
                else torch.where(is_local, local_ids, -1).contiguous()
            ),
        ).to(hidden_states.dtype)


class MuseSpark1xMoE(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        if config.num_shared_experts:
            raise ValueError("Muse Spark supports routed experts only")
        parallel = get_parallel()
        # Experts are split across the TP ranks and reduced after the combine.
        self.tp_size = parallel.tp_size
        tp_rank = parallel.tp_rank
        if config.num_local_experts % self.tp_size:
            raise ValueError("num_local_experts must be divisible by TP size")
        self.num_experts = config.num_local_experts
        num_local = self.num_experts // self.tp_size
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = 1e-15
        self.router_multiplier = config.output_multiplier
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.moe_hidden_size
        self.gate = MuseSpark1xRouter(self.hidden_size, self.num_experts)
        if self.expert_hidden_size != self.hidden_size:
            self.pre_expert_proj = nn.Linear(
                self.hidden_size, self.expert_hidden_size, bias=False
            )
            self.post_expert_proj = nn.Linear(
                self.expert_hidden_size, self.hidden_size, bias=False
            )
            for linear in (self.pre_expert_proj, self.post_expert_proj):
                linear.weight.requires_grad_(False)
        else:
            self.pre_expert_proj = None
            self.post_expert_proj = None
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            self.expert_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
        )
        self.experts = MuseSpark1xExperts(
            num_experts=self.num_experts,
            num_local_experts=num_local,
            local_expert_offset=tp_rank * num_local,
            expert_hidden_size=self.expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            top_k=self.top_k,
        )
        self.experts.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
        )

    def _route(self, router_input: torch.Tensor):
        logits = self.gate(router_input) * self.router_multiplier
        scores = torch.sigmoid(logits.float())
        _, topk_ids = torch.topk(
            scores + self.gate.e_score_correction_bias, self.top_k, dim=-1
        )
        topk_scores = scores.gather(1, topk_ids)
        score_sum = topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores / (score_sum + self.route_eps) * self.route_scale
        return topk_scores, topk_ids.to(torch.int32)

    def forward(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        topk_scores, topk_ids = self._route(router_input)
        expert_input = (
            hidden_states
            if self.pre_expert_proj is None
            else self.pre_expert_proj(hidden_states)
        )
        expert_input = self.pre_expert_norm(expert_input)
        output = self.experts(
            expert_input, topk_scores, topk_ids, self.experts.post_expert_norm
        )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            output = self.post_expert_proj(output)
        return output


class MuseSpark1xDecoderLayer(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.operator_dtype = torch.get_default_dtype()
        self.self_attn = MuseSpark1xAttention(
            config, layer_idx, add_prefix("self_attn", prefix)
        )
        self.input_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
        )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.post_norm_eps, has_weight=False
        )
        self.pre_feedforward_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
        )
        self.post_feedforward_layernorm = MuseSpark1xRMSNormWithInputScale(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
        )
        self.is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"
        self.mlp = (
            MuseSpark1xMoE(config, layer_idx, add_prefix("mlp", prefix))
            if self.is_moe_layer
            else MuseSpark1xMLP(config, add_prefix("mlp", prefix))
        )
        temperature = config.residual_gate_temperature
        self.post_attention_residual_gate = (
            MuseSpark1xNormPreservingResidualConnection(config.hidden_size, temperature)
        )
        self.post_feedforward_residual_gate = (
            MuseSpark1xNormPreservingResidualConnection(config.hidden_size, temperature)
        )

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Takes and returns the FP32 residual stream."""
        attn_input = self.input_layernorm(stream).to(self.operator_dtype)
        attn_output = self.self_attn(positions, attn_input, forward_batch)
        stream = self.post_attention_residual_gate(
            stream, self.post_attn_norm(attn_output)
        )
        ffn_input = self.pre_feedforward_layernorm(stream).to(self.operator_dtype)
        if self.is_moe_layer:
            ffn_output = self.mlp(ffn_input, ffn_input.float())
        else:
            ffn_output = self.mlp(ffn_input)
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(ffn_output)
        )


class MuseSpark1xModel(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, prefix: str):
        super().__init__()
        self.config = config
        self.operator_dtype = torch.get_default_dtype()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.layers = nn.ModuleList(
            MuseSpark1xDecoderLayer(config, i, add_prefix(f"layers.{i}", prefix))
            for i in range(config.num_hidden_layers)
        )
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            gain_center=0.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_norm(self.embed_tokens(input_ids))
        else:
            hidden_states = input_embeds
        stream = hidden_states.float()
        for layer in self.layers:
            stream = layer(positions, stream, forward_batch)
        return self.norm(stream.to(self.operator_dtype))


class MuseSparkModel(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.language_model = MuseSpark1xModel(
            config, add_prefix("language_model", prefix)
        )

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )

    @property
    def layers(self):
        return self.language_model.layers


class MuseSparkTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if quant_config is not None:
            raise ValueError("This Muse Spark port supports BF16 checkpoints only")
        self.config = config
        self.model = MuseSparkModel(config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.language_model.embed_tokens)
        # Logits accumulate in FP32, then scale, then softcap.
        self.logits_processor = LogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.logits_processor.use_fp32_lm_head = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def _load_expert_projection(self, params, name, loaded_weight) -> bool:
        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name
        )
        if match is None:
            return False
        prefix, expert_raw, projection = match.groups()
        param_name = (
            f"{prefix}.{'down_proj' if projection == 'down_proj' else 'gate_up_proj'}"
        )
        param = params.get(param_name)
        if param is None:
            return False
        param.weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=_EXPERT_SHARD_IDS[projection],
            expert_id=int(expert_raw),
        )
        return True

    def _copy_expert_stack(self, param, loaded_weight, name) -> None:
        if loaded_weight.shape[0] == self.config.num_local_experts:
            local = param.shape[0]
            start = get_parallel().tp_rank * local
            loaded_weight = loaded_weight.narrow(0, start, local)
        if param.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        params.update(
            {name: value for name, value in self.named_buffers() if name not in params}
        )
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkv_proj", ".gate_proj", "g"),
        ]
        skipped = 0
        for name, loaded_weight in weights:
            if not (
                name.startswith("model.language_model.") or name == "lm_head.weight"
            ):
                # Vision and audio towers, projectors: text-only serving.
                skipped += 1
                continue
            if should_skip_weight_loading(name, self.config) or name.endswith(
                (".self_attn.q_scale", ".self_attn.k_scale", ".self_attn.v_scale")
            ):
                skipped += 1
                continue
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                # Half-split rotary layout -> interleaved pairs (is_neox_style=False).
                heads = (
                    self.config.num_attention_heads
                    if name.endswith(".q_proj.weight")
                    else self.config.num_key_value_heads
                )
                shape = loaded_weight.shape
                loaded_weight = (
                    loaded_weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
                    .transpose(1, 2)
                    .reshape(shape)
                    .contiguous()
                )
            if self._load_expert_projection(params, name, loaded_weight):
                continue
            handled = False
            for param_pattern, weight_pattern, shard_id in stacked:
                if weight_pattern not in name:
                    continue
                param = params.get(name.replace(weight_pattern, param_pattern))
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{name}' is outside the Muse Spark text model"
                )
            if ".experts." in name and ".post_expert_norm." not in name:
                self._copy_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, MuseSpark1xNormPreservingResidualConnection):
                module.precompute_coefficients()
            elif isinstance(module, MuseSpark1xRMSNorm):
                module.precompute_effective_weight()


class MuseSparkForConditionalGeneration(MuseSparkTextForCausalLM):
    """Muse Spark release checkpoint, served as text (towers are not loaded)."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config, quant_config, prefix)


EntryClass = [MuseSparkForConditionalGeneration, MuseSparkTextForCausalLM]
