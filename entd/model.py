"""Minimal GPT (nanoGPT lineage) that returns logits for ALL positions and the final hidden state."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.resid_dropout = nn.Dropout(c.dropout)
        self.n_head, self.n_embd, self.dropout = c.n_head, c.n_embd, c.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                           dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = LayerNorm(c.n_embd, c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln_2 = LayerNorm(c.n_embd, c.bias)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=LayerNorm(c.n_embd, c.bias),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, return_h=False):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        h = self.transformer.ln_f(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss, (h if return_h else None)

    def unembed_weight(self):
        return self.lm_head.weight   # (V, n_embd)

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in params.items() if p.dim() >= 2]
        no_decay = [p for n, p in params.items() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


class HFCausalLM(nn.Module):
    """Wrapper for HF causal LMs (Pythia etc.) exposing the same interface as GPT."""

    def __init__(self, name, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        self.config = GPTConfig(block_size=self.m.config.max_position_embeddings,
                                vocab_size=self.m.get_output_embeddings().weight.shape[0],
                                n_embd=self.m.config.hidden_size)

    def forward(self, idx, targets=None, return_h=False):
        out = self.m(input_ids=idx, output_hidden_states=return_h)
        logits = out.logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        h = out.hidden_states[-1] if return_h else None   # post final LayerNorm in GPT-NeoX
        return logits, loss, h

    def unembed_weight(self):
        return self.m.get_output_embeddings().weight


def build_model(cfg, meta, device):
    mc = cfg["model"]
    if mc.get("kind", "gpt") == "hf":
        return HFCausalLM(mc["hf_name"]).to(device)
    c = GPTConfig(block_size=cfg["data"]["block_size"], vocab_size=meta["vocab_size"],
                  n_layer=mc["n_layer"], n_head=mc["n_head"], n_embd=mc["n_embd"],
                  dropout=mc["dropout"], bias=mc["bias"])
    return GPT(c).to(device)


def load_checkpoint(cfg, meta, device):
    import os
    if cfg["model"].get("kind", "gpt") == "hf":
        m = HFCausalLM(cfg["model"]["hf_name"]).to(device)
        m.eval()
        return m
    ck = torch.load(os.path.join(cfg["work_dir"], "ckpt.pt"), map_location="cpu")
    m = GPT(GPTConfig(**ck["model_args"]))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(sd)
    m.to(device).eval()
    return m
