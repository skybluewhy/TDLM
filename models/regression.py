import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead,
)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RegressionHead(nn.Module):

    def __init__(
        self, 
        config, 
        out_channels,
        stop_grad=True,
    ):
        super().__init__()

        self.pooler = BertPooler(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.regression_dropout = nn.Dropout(classifier_dropout)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(), 
            nn.Linear(config.hidden_size, out_channels),
            nn.Sigmoid(),
        )

        self.stop_grad = stop_grad

    def forward(self, sequence_output, attn_mask=None):
        if self.stop_grad:
            sequence_output = sequence_output.detach()
        pooled_output = sequence_output.mean(1)
        pooled_output = self.regression_dropout(pooled_output)
        regression_pred = self.regression_head(pooled_output)
        return regression_pred


class RegressionModel(nn.Module):

    def __init__(
        self, 
        config, 
        in_channels, 
        model_channels, 
        out_channels, 
        stop_grad=True,
    ):
        super().__init__()

        self.model_channels = model_channels
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )
        
        self.input_up_proj = nn.Linear(in_channels, config.hidden_size)
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.regression_transformers = BertEncoder(config)
        self.regression_head = RegressionHead(config, out_channels)

        self.stop_grad = stop_grad

    def forward(self, sequence_embeds, timesteps, attn_mask=None):
        if self.stop_grad:
            sequence_embeds = sequence_embeds.detach()

        t_feats = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_feats)
        
        emb_x = self.input_up_proj(sequence_embeds)
        seq_length = sequence_embeds.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        
        if attn_mask is None:
            attn_mask = torch.ones_like(sequence_embeds[...,0])
        attn_mask = attn_mask[:,None,None,:]
        
        h = self.regression_transformers(emb_inputs, attention_mask=attn_mask).last_hidden_state
        return self.regression_head(h)
