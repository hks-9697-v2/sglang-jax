import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class Gemma4MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.gelu

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Gemma4Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        max_position_embeddings: int,
        attention_bias: bool,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_type = "full_attention"
        if hasattr(config, "layer_types") and layer_id < len(config.layer_types):
            self.layer_type = config.layer_types[layer_id]

        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = getattr(config, "sliding_window", 0) if self.is_sliding else 0

        rope_parameters = getattr(config, "rope_parameters", {})
        if self.layer_type in rope_parameters:
            rope_params = rope_parameters[self.layer_type]
            rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
            rope_scaling = rope_params.get("rope_scaling", getattr(config, "rope_scaling", None))
            default_prop = 0.25 if not self.is_sliding else 1.0
            rope_proportion = rope_params.get("partial_rotary_factor", default_prop)
        else:
            rope_theta = getattr(config, "rope_local_base_freq", 10000.0) if self.is_sliding else getattr(config, "rope_theta", 1000000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            rope_proportion = 0.25 if not self.is_sliding else 1.0

        if not self.is_sliding:
            self.head_dim = getattr(config, "global_head_dim", getattr(config, "head_dim", self.hidden_size // self.num_heads))
        else:
            self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.use_k_eq_v = ((not self.is_sliding) and getattr(config, "attention_k_eq_v", False))
        if self.use_k_eq_v:
            self.num_kv_heads = getattr(config, "num_global_key_value_heads", getattr(config, "num_key_value_heads", self.num_heads))
        else:
            self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        self.q_head_num = self.num_heads
        self.kv_head_num = self.num_kv_heads
        self.mesh = mesh

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.q_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )

        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.k_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        if self.use_k_eq_v:
            self.v_proj = None
        else:
            self.v_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
                use_bias=attention_bias,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
                scope_name="v_proj",
            )
        self.v_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, use_scale=False, scope_name="v_norm"
        )

        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=None,
            dtype=dtype,
            partial_rotary_factor=rope_proportion,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        if self.v_proj is None:
            v = k
        else:
            v, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        v = self.v_norm(v)

        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma4DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 256000)
        attention_bias = getattr(config, "attention_bias", False)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=dtype))

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.self_attn = Gemma4Attention(
            config=config,
            layer_id=layer_id,
            max_position_embeddings=max_position_embeddings,
            attention_bias=attention_bias,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="pre_feedforward_layernorm",
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_feedforward_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        layer_callback_flag = []
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        layer_norm_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "input_layernorm_output", "INPUT_LAYERNORM", self.layer_id
        )
        layer_callback_flag.append(layer_norm_callback_flag)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        attn_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "self_attn_output", "SELF_ATTN", self.layer_id
        )
        layer_callback_flag.append(attn_callback_flag)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states += residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states += residual

        hidden_states = hidden_states * self.layer_scalar.value

        mlp_callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "mlp_output", "MLP", self.layer_id
        )
        layer_callback_flag.append(mlp_callback_flag)

        return hidden_states, residual, kv_fused, layer_callback_flag


class Gemma4Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Gemma4DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=getattr(config, "rms_norm_eps", 1e-6),
            param_dtype=dtype,
            scope_name="norm",
        )
        self.hidden_size = config.hidden_size
        self.layers_to_capture = []

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        hidden_states *= jnp.array([self.hidden_size**0.5], dtype=hidden_states.dtype)

        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual, kv_fused, callback_flag = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class Gemma4ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype
        logger.info("Gemma4ForCausalLM config dtype: %s", self.dtype)
        self.model = Gemma4Model(self.config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", 0.0),
            mesh=self.mesh,
        )
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_gemma4_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.layer_scalar.value = jax.device_put(jnp.ones((1,), dtype=self.dtype), jax.sharding.NamedSharding(self.mesh, P()))

        if hasattr(self, "lm_head"):
            if isinstance(self.lm_head.embedding.value, jax.ShapeDtypeStruct):
                logger.info("Tying lm_head weights to embed_tokens (lm_head not in safetensors)")
                self.lm_head.embedding = self.model.embed_tokens.embedding

        logger.info("Gemma4 weights loaded successfully!")

    def _create_gemma4_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if hasattr(self, "lm_head"):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        multimodal_mappings = {f"language_model.{k}": v for k, v in mappings.items()}
        mappings.update(multimodal_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        layer_type = "full_attention"
        if hasattr(self.config, "layer_types") and layer_idx < len(self.config.layer_types):
            layer_type = self.config.layer_types[layer_idx]
        is_sliding = layer_type == "sliding_attention"
        use_k_eq_v = ((not is_sliding) and getattr(self.config, "attention_k_eq_v", False))

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.pre_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.pre_feedforward_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_feedforward_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_feedforward_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if not use_k_eq_v:
            mappings[f"{prefix}.self_attn.v_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            )

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            if not use_k_eq_v:
                bias_mappings[f"{prefix}.self_attn.v_proj.bias"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                )
            mappings.update(bias_mappings)

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.config, "tie_word_embeddings", True):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, layers_kv_fused, layers_callback_flag, None


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    pass


EntryClass = [Gemma4ForCausalLM, Gemma4ForConditionalGeneration]
