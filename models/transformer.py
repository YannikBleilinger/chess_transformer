import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel


class Config(PretrainedConfig):
    model_type = "my_chessformer"
    vocab_size = 4272
    n_layers = 12
    n_heads = 16
    n_head_kv_divider = 4
    dim = 1024
    ffn_dim_multiplier = 2
    dropout = 0.1
    norm_eps = 1e-6

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class SelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = max(1, self.n_heads // config.n_head_kv_divider)
        self.dropout = config.dropout

        # GQA
        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)  # value up
        self.wo = nn.Linear(config.dim, config.dim, bias=False)  # value down


    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # make heads be a batch dim
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # use PyTorch implementation of FlashAttention (if available)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False, enable_gqa=True, dropout_p=self.dropout)
        output = output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        proj = self.wo(output)

        return proj


# FFN
class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden_dim = config.dim * config.ffn_dim_multiplier
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        # implementation of the swiglu function
        x_ffn = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.dropout(x_ffn)

# Transformer block
class EncoderBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // config.n_heads

        self.attention = SelfAttention(config)
        self.ffn = FeedForward(config)

        # normalization before self attention
        self.att_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # normalization before ffn
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor):
        h = x + self.attention(self.att_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out


# Transformer
class Transformer(PreTrainedModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.input_proj = nn.Linear(7, config.dim)
        self.pos_embedding = nn.Embedding(65, config.dim)
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(self.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.head = nn.Linear(config.dim, 4272)
        self.init_weights()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):

        batch_size, seq_len, _ = tokens.shape
        tokens = self.input_proj(tokens)  # (B, seq_len, config.dim)
        # encode position
        positons = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        h = tokens + self.pos_embedding(positons)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            log_probs = F.log_softmax(logits[:, 0, :], dim=-1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean')
        return {"loss": loss, "logits": logits}


