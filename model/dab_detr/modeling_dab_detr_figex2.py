# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor, nn


from transformers.activations import ACT2FN
from transformers import initialization as init
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    logging,
)


try:
    from transformers.backbone_utils import load_backbone
except ImportError:
    from transformers.utils.backbone_utils import load_backbone
from transformers.models.dab_detr.configuration_dab_detr import DabDetrConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the Conditional DETR decoder. This class adds one attribute to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations, i.e. the output
    of each decoder layer, each of them gone through a layernorm.
    """
)

class DabDetrDecoderOutput(BaseModelOutputWithCrossAttentions):


    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    reference_points: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the Conditional DETR encoder-decoder model. This class adds one attribute to
    Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder
    layer, each of them gone through a layernorm.
    """
)

class DabDetrModelOutput(Seq2SeqModelOutput):


    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    reference_points: Optional[tuple[torch.FloatTensor]] = None
    text_attention: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`DabDetrForObjectDetection`].
    """
)

class DabDetrObjectDetectionOutput(ModelOutput):


    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[list[dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    text_attention: Optional[torch.FloatTensor] = None


class DabDetrFrozenBatchNorm2d(nn.Module):


    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):


        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


def replace_batch_norm(model):


    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DabDetrFrozenBatchNorm2d(module.num_features)

            if module.weight.device != torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DabDetrConvEncoder(nn.Module):


    def __init__(self, config: DabDetrConfig):
        super().__init__()

        self.config = config
        backbone = load_backbone(config)


        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):

        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:

            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


class DabDetrConvModel(nn.Module):


    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):

        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:

            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class DabDetrSinePositionEmbedding(nn.Module):


    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size / 2
        self.temperature_height = config.temperature_height
        self.temperature_width = config.temperature_width
        scale = config.sine_position_embedding_scale
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale


        dim_tx = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)

        dim_tx //= 2
        dim_tx.mul_(2 / self.embedding_dim)
        dim_tx.copy_(self.temperature_width**dim_tx)
        pos_x = x_embed[:, :, :, None] / dim_tx


        dim_ty = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)

        dim_ty //= 2
        dim_ty.mul_(2 / self.embedding_dim)
        dim_ty.copy_(self.temperature_height**dim_ty)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def gen_sine_position_embeddings(pos_tensor, hidden_size=256):


    scale = 2 * math.pi
    dim = hidden_size // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos.to(pos_tensor.dtype)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DetrAttention(nn.Module):


    def __init__(
        self,
        config: DabDetrConfig,
        bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.encoder_attention_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        batch_size, q_len, embed_dim = hidden_states.size()

        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = hidden_states + object_queries

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states_original)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, q_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class DabDetrAttention(nn.Module):


    def __init__(self, config: DabDetrConfig, bias: bool = True, is_cross: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size * 2 if is_cross else config.hidden_size
        self.output_dim = config.hidden_size
        self.attention_heads = config.decoder_attention_heads
        self.attention_dropout = config.attention_dropout
        self.attention_head_dim = self.embed_dim // self.attention_heads
        if self.attention_head_dim * self.attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `attention_heads`:"
                f" {self.attention_heads})."
            )

        self.values_head_dim = self.output_dim // self.attention_heads
        if self.values_head_dim * self.attention_heads != self.output_dim:
            raise ValueError(
                f"output_dim must be divisible by attention_heads (got `output_dim`: {self.output_dim} and `attention_heads`: {self.attention_heads})."
            )
        self.scaling = self.attention_head_dim**-0.5
        self.output_proj = nn.Linear(self.output_dim, self.output_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:


        batch_size, q_len, _ = hidden_states.size()


        query_states = hidden_states * self.scaling
        query_states = query_states.view(batch_size, -1, self.attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.attention_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.attention_heads, self.values_head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (batch_size, self.attention_heads, q_len, self.values_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.attention_heads, q_len, self.values_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, q_len, self.output_dim)
        attn_output = self.output_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class DabDetrDecoderLayerSelfAttention(nn.Module):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.dropout = config.dropout
        self.self_attn_query_content_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_query_pos_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_key_content_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_key_pos_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn = DabDetrAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        residual = hidden_states
        query_content = self.self_attn_query_content_proj(hidden_states)
        query_pos = self.self_attn_query_pos_proj(query_position_embeddings)
        key_content = self.self_attn_key_content_proj(hidden_states)
        key_pos = self.self_attn_key_pos_proj(query_position_embeddings)
        value = self.self_attn_value_proj(hidden_states)

        query = query_content + query_pos
        key = key_content + key_pos

        hidden_states, attn_weights = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            key_states=key,
            value_states=value,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        return hidden_states, attn_weights


class DabDetrDecoderLayerCrossAttention(nn.Module):
    def __init__(self, config: DabDetrConfig, is_first: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        self.cross_attn_query_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_value_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_sine_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_attention_heads = config.decoder_attention_heads
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = DabDetrAttention(config, is_cross=True)

        self.keep_query_pos = config.keep_query_pos

        if not self.keep_query_pos and not is_first:
            self.cross_attn_query_pos_proj = None

        self.is_first = is_first
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        query_content = self.cross_attn_query_content_proj(hidden_states)
        key_content = self.cross_attn_key_content_proj(encoder_hidden_states)
        value = self.cross_attn_value_proj(encoder_hidden_states)

        batch_size, num_queries, n_model = query_content.shape
        _, height_width, _ = key_content.shape

        key_pos = self.cross_attn_key_pos_proj(object_queries)


        if self.is_first or self.keep_query_pos:
            query_pos = self.cross_attn_query_pos_proj(query_position_embeddings)
            query = query_content + query_pos
            key = key_content + key_pos
        else:
            query = query_content
            key = key_content

        query = query.view(
            batch_size, num_queries, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        query_sine_embed = self.cross_attn_query_pos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(
            batch_size, num_queries, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        query = torch.cat([query, query_sine_embed], dim=3).view(batch_size, num_queries, n_model * 2)
        key = key.view(batch_size, height_width, self.decoder_attention_heads, n_model // self.decoder_attention_heads)
        key_pos = key_pos.view(
            batch_size, height_width, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        key = torch.cat([key, key_pos], dim=3).view(batch_size, height_width, n_model * 2)


        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states=query,
                attention_mask=encoder_attention_mask,
                key_states=key,
                value_states=value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        return hidden_states, cross_attn_weights


class DabDetrDecoderLayerFFN(nn.Module):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, hidden_size)
        self.activation_fn = ACT2FN[config.activation_function]
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.keep_query_pos = config.keep_query_pos

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class DabDetrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DetrAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.hidden_size, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.hidden_size)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor,
        output_attentions: Optional[bool] = None,
    ):


        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class DabDetrDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DabDetrConfig, is_first: bool = False):
        super().__init__()
        self.self_attn = DabDetrDecoderLayerSelfAttention(config)
        self.cross_attn = DabDetrDecoderLayerCrossAttention(config, is_first)
        self.mlp = DabDetrDecoderLayerFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):


        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            query_position_embeddings=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_position_embeddings=query_position_embeddings,
            object_queries=object_queries,
            encoder_attention_mask=encoder_attention_mask,
            query_sine_embed=query_sine_embed,
            output_attentions=output_attentions,
        )

        hidden_states = self.mlp(hidden_states=hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class DabDetrMLP(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, input_tensor):
        for i, layer in enumerate(self.layers):
            input_tensor = nn.functional.relu(layer(input_tensor)) if i < self.num_layers - 1 else layer(input_tensor)
        return input_tensor


def _init_mha_in_proj(module: nn.MultiheadAttention, xavier_gain: float | None):


    gain = 1.0 if (xavier_gain is None) else float(xavier_gain)

    if hasattr(module, "q_proj_weight") and module.q_proj_weight is not None:
        init.xavier_uniform_(module.q_proj_weight, gain=gain)
        init.xavier_uniform_(module.k_proj_weight, gain=gain)
        init.xavier_uniform_(module.v_proj_weight, gain=gain)
        if getattr(module, "in_proj_bias", None) is not None:
            init.zeros_(module.in_proj_bias)
    else:

        init.xavier_uniform_(module.in_proj_weight, gain=gain)
        if module.in_proj_bias is not None:
            init.zeros_(module.in_proj_bias)

@auto_docstring
class DabDetrPreTrainedModel(PreTrainedModel):
    config: DabDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DabDetrConvEncoder", r"DabDetrEncoderLayer", r"DabDetrDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if getattr(module, "_figex2_zero_init", False):
            if isinstance(module, nn.MultiheadAttention):
                init.zeros_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    init.zeros_(module.in_proj_bias)
                init.zeros_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    init.zeros_(module.out_proj.bias)
            elif isinstance(module, nn.Linear):
                init.zeros_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        elif isinstance(module, DabDetrMHAttentionMap):
            init.zeros_(module.k_linear.bias)
            init.zeros_(module.q_linear.bias)
            init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)


        elif isinstance(module, nn.MultiheadAttention):
            _init_mha_in_proj(module, xavier_std)

        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])

        elif isinstance(module, DabDetrForObjectDetection):
            init.constant_(module.bbox_predictor.layers[-1].weight, 0)
            init.constant_(module.bbox_predictor.layers[-1].bias, 0)


            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            init.constant_(module.class_embed.bias, bias_value)

        elif isinstance(module, nn.PReLU):
            module.reset_parameters()


class DabDetrEncoder(DabDetrPreTrainedModel):


    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.query_scale = DabDetrMLP(config.hidden_size, config.hidden_size, config.hidden_size, 2)
        self.layers = nn.ModuleList([DabDetrEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size) if config.normalize_before else None
        self.gradient_checkpointing = False


        self.post_init()

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        object_queries,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            pos_scales = self.query_scale(hidden_states)

            scaled_object_queries = object_queries * pos_scales

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                object_queries=scaled_object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.norm:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DabDetrDecoder(DabDetrPreTrainedModel):


    def __init__(self, config: DabDetrConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.num_layers = config.decoder_layers
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [DabDetrDecoderLayer(config, is_first=(layer_id == 0)) for layer_id in range(config.decoder_layers)]
        )

        self.hidden_size = config.hidden_size
        self.layernorm = nn.LayerNorm(self.hidden_size)


        self.query_scale = DabDetrMLP(self.hidden_size, self.hidden_size, self.hidden_size, 2)

        self.ref_point_head = DabDetrMLP(
            config.query_dim // 2 * self.hidden_size, self.hidden_size, self.hidden_size, 2
        )

        self.bbox_embed = None


        self.ref_anchor_head = DabDetrMLP(self.hidden_size, self.hidden_size, 2, 2)


        self.post_init()

    def forward(
        self,
        inputs_embeds,
        encoder_hidden_states,
        memory_key_padding_mask,
        object_queries,
        query_position_embeddings,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds


        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        intermediate = []
        reference_points = query_position_embeddings.sigmoid()
        ref_points = [reference_points]


        if encoder_hidden_states is not None and memory_key_padding_mask is not None:
            memory_key_padding_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=memory_key_padding_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        for layer_id, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            obj_center = reference_points[..., : self.config.query_dim]
            query_sine_embed = gen_sine_position_embeddings(obj_center, self.hidden_size)
            query_pos = self.ref_point_head(query_sine_embed)


            pos_transformation = 1 if layer_id == 0 else self.query_scale(hidden_states)


            query_sine_embed = query_sine_embed[..., : self.hidden_size] * pos_transformation


            reference_anchor_size = self.ref_anchor_head(hidden_states).sigmoid()
            query_sine_embed[..., self.hidden_size // 2 :] *= (
                reference_anchor_size[..., 0] / obj_center[..., 2]
            ).unsqueeze(-1)
            query_sine_embed[..., : self.hidden_size // 2] *= (
                reference_anchor_size[..., 1] / obj_center[..., 3]
            ).unsqueeze(-1)

            layer_outputs = decoder_layer(
                hidden_states,
                None,
                object_queries,
                query_pos,
                query_sine_embed,
                encoder_hidden_states,
                encoder_attention_mask=memory_key_padding_mask,
                output_attentions=output_attentions,
            )


            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                new_reference_points = self.bbox_embed(hidden_states)

                new_reference_points[..., : self.config.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points[..., : self.config.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            intermediate.append(self.layernorm(hidden_states))

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)


        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output_intermediate_hidden_states = torch.stack(intermediate)
        output_reference_points = torch.stack(ref_points)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                    output_intermediate_hidden_states,
                    output_reference_points,
                ]
                if v is not None
            )
        return DabDetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=output_intermediate_hidden_states,
            reference_points=output_reference_points,
        )


@auto_docstring(
    custom_intro="""
    The bare DAB-DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states, intermediate hidden states, reference points, output coordinates without any specific head on top.
    """
)
class DabDetrModel(DabDetrPreTrainedModel):
    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.auxiliary_loss = config.auxiliary_loss


        det_in = getattr(config, "det_feat_dim", 4096)
        txt_in = getattr(config, "text_feat_dim", 4096)
        self.text_guidance_enabled = True

        H = self.config.hidden_size
        heads = config.decoder_attention_heads

        self.det_proj = nn.Linear(det_in, H)
        self.text_proj = nn.Linear(txt_in, H)
        self.det_proj._figex2_zero_init = True
        self.text_proj._figex2_zero_init = True


        self.q_det_attn = nn.MultiheadAttention(embed_dim=H, num_heads=heads, batch_first=True)
        self.q_det_attn._figex2_zero_init = True
        self.q_det_ln = nn.LayerNorm(H)


        self.q_txt_attn = nn.MultiheadAttention(embed_dim=H, num_heads=heads, batch_first=True)
        self.q_txt_attn._figex2_zero_init = True
        self.q_txt_ln = nn.LayerNorm(H)


        self.backbone = DabDetrConvEncoder(config)
        object_queries = DabDetrSinePositionEmbedding(config)

        self.query_refpoint_embeddings = nn.Embedding(config.num_queries, config.query_dim)
        self.random_refpoints_xy = config.random_refpoints_xy
        if self.random_refpoints_xy:
            self.query_refpoint_embeddings.weight.data[:, :2].uniform_(0, 1)
            self.query_refpoint_embeddings.weight.data[:, :2] = inverse_sigmoid(
                self.query_refpoint_embeddings.weight.data[:, :2]
            )
            self.query_refpoint_embeddings.weight.data[:, :2].requires_grad = False


        self.input_projection = nn.Conv2d(
            self.backbone.intermediate_channel_sizes[-1], config.hidden_size, kernel_size=1
        )
        self.backbone = DabDetrConvModel(self.backbone, object_queries)

        self.encoder = DabDetrEncoder(config)
        self.decoder = DabDetrDecoder(config)


        self.hidden_size = config.hidden_size
        self.num_queries = config.num_queries

        self.num_patterns = config.num_patterns
        if not isinstance(self.num_patterns, int):
            logger.warning(f"num_patterns should be int but {type(self.num_patterns)}")
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.hidden_size)

        self.aux_loss = config.auxiliary_loss


        self.post_init()
        self._zero_init_query_text_fusion()

    def _zero_init_query_text_fusion(self):
        for module in (self.det_proj, self.text_proj):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        for attn in (self.q_det_attn, self.q_txt_attn):
            nn.init.zeros_(attn.in_proj_weight)
            if attn.in_proj_bias is not None:
                nn.init.zeros_(attn.in_proj_bias)
            nn.init.zeros_(attn.out_proj.weight)
            if attn.out_proj.bias is not None:
                nn.init.zeros_(attn.out_proj.bias)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


        det_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,


    ) -> Union[tuple[torch.FloatTensor], DabDetrModelOutput]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)


        features, object_queries_list = self.backbone(pixel_values, pixel_mask)


        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        flattened_mask = mask.flatten(1)


        projected_feature_map = self.input_projection(feature_map)


        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        reference_position_embeddings = self.query_refpoint_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)


        use_text_guidance = (
            self.text_guidance_enabled
            and (det_feats is not None)
            and (text_feats is not None)
        )
        if use_text_guidance:
            if det_feats.dim() == 2:
                det_feats = det_feats.unsqueeze(1)

            if text_feats.dim() == 2:
                text_feats = text_feats.unsqueeze(1)

            det_h = self.det_proj(det_feats)
            text_h = self.text_proj(text_feats)
        else:
            det_h = None
            text_h = None


        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )


        num_queries = reference_position_embeddings.shape[1]
        if self.num_patterns == 0:
            queries = torch.zeros(batch_size, num_queries, self.hidden_size, device=device)
        else:
            queries = (
                self.patterns.weight[:, None, None, :]
                .repeat(1, self.num_queries, batch_size, 1)
                .flatten(0, 1)
                .permute(1, 0, 2)
            )
            reference_position_embeddings = reference_position_embeddings.repeat(
                1, self.num_patterns, 1
            )


        if use_text_guidance:
            upd, _ = self.q_det_attn(
                query=queries,
                key=det_h,
                value=det_h,
            )
            queries = self.q_det_ln(queries + upd)

            upd, text_attn = self.q_txt_attn(
                query=queries,
                key=text_h,
                value=text_h,
                key_padding_mask=text_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            queries = self.q_txt_ln(queries + upd)
            text_attention = text_attn.mean(dim=1)
        else:
            text_attention = None


        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            query_position_embeddings=reference_position_embeddings,
            object_queries=object_queries,
            encoder_hidden_states=encoder_outputs[0],
            memory_key_padding_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:

            output = (decoder_outputs[0],)
            reference_points = decoder_outputs[-1]
            intermediate_hidden_states = decoder_outputs[-2]


            if output_hidden_states and output_attentions:
                output += (
                    decoder_outputs[1],
                    decoder_outputs[2],
                    decoder_outputs[3],
                    encoder_outputs[0],
                    encoder_outputs[1],
                    encoder_outputs[2],
                )
            elif output_hidden_states:

                output += (
                    decoder_outputs[1],
                    encoder_outputs[0],
                    encoder_outputs[1],
                )
            elif output_attentions:

                output += (
                    decoder_outputs[1],
                    decoder_outputs[2],
                    encoder_outputs[1],
                )

            output += (intermediate_hidden_states, reference_points)

            return output

        reference_points = decoder_outputs.reference_points
        intermediate_hidden_states = decoder_outputs.intermediate_hidden_states

        return DabDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states if output_hidden_states else None,
            decoder_attentions=decoder_outputs.attentions if output_attentions else None,
            cross_attentions=decoder_outputs.cross_attentions if output_attentions else None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if output_hidden_states else None,
            encoder_hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            encoder_attentions=encoder_outputs.attentions if output_attentions else None,
            intermediate_hidden_states=intermediate_hidden_states,
            reference_points=reference_points,
            text_attention=text_attention,
        )


class DabDetrMHAttentionMap(nn.Module):


    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


@auto_docstring(
    custom_intro="""
    DAB_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """
)
class DabDetrForObjectDetection(DabDetrPreTrainedModel):

    _tied_weights_keys = {"model.decoder.bbox_embed": "bbox_predictor"}

    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.config = config
        self.auxiliary_loss = config.auxiliary_loss
        self.query_dim = config.query_dim

        self.model = DabDetrModel(config)


        self.class_embed = nn.Linear(config.hidden_size, config.num_labels)


        self.bbox_predictor = DabDetrMLP(config.hidden_size, config.hidden_size, 4, 3)


        self.model.decoder.bbox_embed = self.bbox_predictor


        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._reload_tied_bbox_embed_from_checkpoint(pretrained_model_name_or_path)
        return model

    def _reload_tied_bbox_embed_from_checkpoint(self, pretrained_model_name_or_path) -> None:
        ckpt_dir = Path(str(pretrained_model_name_or_path))
        safetensors_path = ckpt_dir / "model.safetensors"
        if not safetensors_path.exists():
            return
        try:
            from safetensors.torch import load_file

            state = load_file(str(safetensors_path), device=str(self.device))
        except Exception:
            return

        current = self.state_dict()
        to_load = {}
        for key in current:
            if key.startswith("model.decoder.bbox_embed."):
                source = state.get(key)
                if source is not None and source.shape == current[key].shape:
                    to_load[key] = source
        if to_load:
            self.load_state_dict(to_load, strict=False)


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):


        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


        det_feats: Optional[torch.Tensor] = None,
        text_feats: Optional[torch.Tensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,


    ) -> Union[tuple[torch.FloatTensor], DabDetrObjectDetectionOutput]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        model_outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            det_feats=det_feats,
            text_feats=text_feats,
            text_key_padding_mask=text_key_padding_mask,

        )

        reference_points = model_outputs.reference_points if return_dict else model_outputs[-1]
        intermediate_hidden_states = model_outputs.intermediate_hidden_states if return_dict else model_outputs[-2]


        logits = self.class_embed(intermediate_hidden_states[-1])

        reference_before_sigmoid = inverse_sigmoid(reference_points)
        bbox_with_refinement = self.bbox_predictor(intermediate_hidden_states)
        bbox_with_refinement[..., : self.query_dim] += reference_before_sigmoid
        outputs_coord = bbox_with_refinement.sigmoid()

        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class = None
            if self.config.auxiliary_loss:
                outputs_class = self.class_embed(intermediate_hidden_states)
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + model_outputs
            else:
                output = (logits, pred_boxes) + model_outputs

            return ((loss, loss_dict) + output) if loss is not None else output[:-2]

        return DabDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=model_outputs.last_hidden_state,
            decoder_hidden_states=model_outputs.decoder_hidden_states if output_hidden_states else None,
            decoder_attentions=model_outputs.decoder_attentions if output_attentions else None,
            cross_attentions=model_outputs.cross_attentions if output_attentions else None,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state if output_hidden_states else None,
            encoder_hidden_states=model_outputs.encoder_hidden_states if output_hidden_states else None,
            encoder_attentions=model_outputs.encoder_attentions if output_attentions else None,
            text_attention=model_outputs.text_attention,
        )


__all__ = [
    "DabDetrForObjectDetection",
    "DabDetrModel",
    "DabDetrPreTrainedModel",
]
