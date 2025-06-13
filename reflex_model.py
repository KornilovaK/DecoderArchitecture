import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn import flash_attn_func

from transformers import GPT2LMHeadModel


# RMSNorm instead of LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6, bias=False):
        super().__init__()
        self.weight =   nn.Parameter(torch.ones(ndim))
        self.bias   =   nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps    =   eps

    def forward(self, x):
        rms       =   torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True))
        x_normed  =   x / (rms + self.eps)
        return x_normed * self.weight + self.bias if self.bias else x_normed * self.weight

                
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn         =   nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj         =   nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout   =   nn.Dropout(config.dropout)
        self.resid_dropout  =   nn.Dropout(config.dropout)
        self.n_head         =   config.n_head
        self.n_embd         =   config.n_embd
        self.dropout        =   config.dropout

    def forward(self, x, prevs):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        cur_kv = (k, v)
        sa_h = self.n_head - len(prevs)
        # self_attention = F.scaled_dot_product_attention(q[:, :sa_h, :, :], k[:, :sa_h, :, :], v[:, :sa_h, :, :], is_causal=True, dropout_p=self.dropout if self.training else 0)
        self_attention = flash_attn_func(q[:, :sa_h, :, :], k[:, :sa_h, :, :], v[:, :sa_h, :, :], causal=True, dropout_p=self.dropout if self.training else 0)

        q = q[:, sa_h:, :, :]        
        k = torch.cat([cur_k[:, (sa_h+i), :, :].unsqueeze(1) for i, (cur_k, cur_v) in enumerate(prevs)], dim=1)
        v = torch.cat([cur_v[:, (sa_h+i), :, :].unsqueeze(1) for i, (cur_k, cur_v) in enumerate(prevs)], dim=1)
        
        # cross_attention = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout if self.training else 0)
        cross_attention = flash_attn_func(q, k, v, causal=False, dropout_p=self.dropout if self.training else 0)

        y = torch.cat([self_attention, cross_attention], dim=1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, cur_kv


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj       =   nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation =   nn.SiLU()

    def forward(self, hidden_states):
        hidden_states       =   self.proj(hidden_states)
        hidden_states, gate =   hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    =   nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act     =   SwiGLU(4 * config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  =   nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout =   nn.Dropout(config.dropout)

    def forward(self, x):
        x   =   self.c_fc(x)
        x   =   self.act(x)
        x   =   self.c_proj(x)
        x   =   self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1   =   RMSNorm(config.n_embd)
        self.attn   =   Attention(config)
        self.ln_2   =   RMSNorm(config.n_embd)
        self.mlp    =   MLP(config)

    def forward(self, x, prev_kvs=None):
        normed_x1    =   self.ln_1(x)                    # PreNorm 1
        attn_out, kv =   self.attn(normed_x1, prev_kvs)  # attention block
        x            =   x + attn_out                    # Residual connection + attention block
        normed_x2    =   self.ln_2(x)                    # PreNorm 2
        ffn          =   self.mlp(normed_x2)             # FFN
        x            =   x + ffn                         # Residual connection + FFN
        return x, kv                                     # return layer output + current kv hiddens


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.transformer = nn.ModuleDict(dict(
            wte   =   nn.Embedding(config.vocab_size, config.n_embd),
            wpe   =   nn.Embedding(config.block_size, config.n_embd),
            drop  =   nn.Dropout(config.dropout),
            h     =   nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f  =   RMSNorm(config.n_embd, bias=config.bias),
        ))

        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self._init_from_pretrained_weights()

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_from_pretrained_weights(self):
        pretrained_model = GPT2LMHeadModel.from_pretrained(
            self.config.pretrained_model_path,
            local_files_only=self.config.local_files_only,
            trust_remote_code=True
        )
        
        pretrained_state = pretrained_model.state_dict()
        current_state = self.state_dict()
        name_mapping = {
            "transformer.wte.weight":  "transformer.wte.weight",
            "lm_head.weight":          "lm_head.weight",
            "transformer.wpe.weight":  "transformer.wpe.weight",
            "transformer.ln_f.weight": "transformer.ln_f.weight",
            "transformer.ln_f.bias":   "transformer.ln_f.bias",
        }
        
        for i in range(min(len(self.transformer.h), pretrained_model.config.n_layer)):
            prefix = f"transformer.h.{i}."
            name_mapping.update({
                f"{prefix}ln_1.weight":        f"transformer.h.{i}.ln_1.weight",
                f"{prefix}ln_1.bias":          f"transformer.h.{i}.ln_1.bias",
                f"{prefix}ln_2.weight":        f"transformer.h.{i}.ln_2.weight",
                f"{prefix}ln_2.bias":          f"transformer.h.{i}.ln_2.bias",
                
                f"{prefix}attn.c_attn.weight": f"transformer.h.{i}.attn.c_attn.weight",
                f"{prefix}attn.c_attn.bias":   f"transformer.h.{i}.attn.c_attn.bias",
                f"{prefix}attn.c_proj.weight": f"transformer.h.{i}.attn.c_proj.weight",
                f"{prefix}attn.c_proj.bias":   f"transformer.h.{i}.attn.c_proj.bias",
                
                f"{prefix}mlp.c_fc.weight":    f"transformer.h.{i}.mlp.c_fc.weight",
                f"{prefix}mlp.c_fc.bias":      f"transformer.h.{i}.mlp.c_fc.bias",
                f"{prefix}mlp.c_proj.weight":  f"transformer.h.{i}.mlp.c_proj.weight",
                f"{prefix}mlp.c_proj.bias":    f"transformer.h.{i}.mlp.c_proj.bias",
            })
        
        for pt_name, our_name in name_mapping.items():
            if our_name in current_state and pt_name in pretrained_state:
                pt_param = pretrained_state[pt_name]
                our_param = current_state[our_name]

                if 'weight' in pt_name and any(x in pt_name for x in ['c_attn', 'c_proj', 'c_fc']):
                    pt_param = pt_param.t()
                
                if our_param.shape == pt_param.shape:
                    our_param.data.copy_(pt_param)
                    print(f"Copied: {pt_name} -> {our_name}")
                    
                elif our_param.dim() == pt_param.dim():
                    min_shape = [min(s1, s2) for s1, s2 in zip(our_param.shape, pt_param.shape)]
                    slices = tuple(slice(0, s) for s in min_shape)
                    our_param.data[slices].copy_(pt_param[slices])
                    print(f"Partially copied: {pt_name} -> {our_name} (shape {pt_param.shape} -> {our_param.shape})")
                else:
                    print(f"Skipped: {pt_name} (shape mismatch {pt_param.shape} vs {our_param.shape})")
        
        for name, param in self.named_parameters():
            if name not in name_mapping.values():
                print(f"Initializing new parameter: {name}")
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        return self
        
    
    def forward(self, idx, targets=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        prevs = []
        for i, block in enumerate(self.transformer.h):
            x, prev = block(x, prevs)
            prevs.append(prev)

        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, beam_sample=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                try:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cumulative_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    logits[sorted_indices[remove_mask]] = -float('Inf')
                except Exception as e:
                    print(f"FIX TOP-P: {e}")
                    
            probs = F.softmax(logits, dim=-1)
            
            if beam_sample:
                print("Not implemented yet")
                
            idx_next = torch.multinomial(probs, num_samples=1) # if beam_sample == True: torch.multinomial(probs, num_samples=num_samples)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params