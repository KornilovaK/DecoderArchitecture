import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-6)

class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head # n_heads = 8
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.model_type = config.model_type

    def forward(self, x, prevs):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        if self.model_type == 'original' or (self.model_type == 'reflex' and len(prevs) <= 1): # len(prevs) == 0):
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else: 
            # # for all_layer reflex attention: n_sa_heads = n_heads - len(prev_head) -> 8; 7+1; 6+1+1; 5+1+1+1; 4+1+1+1+1; 3+1+1+1+1+1
            # sa_h = self.n_head - len(prevs)
            # q_sa, k_sa, v_sa = q[:, :sa_h, :, :], k[:, :sa_h, :, :], v[:, :sa_h, :, :]
            # self_attention = torch.nn.functional.scaled_dot_product_attention(q_sa, k_sa, v_sa, attn_mask=None,
            #                                                                 dropout_p=self.dropout if self.training else 0,
            #                                                                 is_causal=True)
            # qkv = [self_attention]
            # for i, prev in enumerate(prevs):
            #     q_ca, k_ca, v_ca = q[:, (sa_h+i), :, :].unsqueeze(1), prev[0][:, (sa_h+i), :, :].unsqueeze(1), prev[1][:, (sa_h+i), :, :].unsqueeze(1)
            #     cross_attention = torch.nn.functional.scaled_dot_product_attention(q_ca, k_ca, v_ca, attn_mask=None,
            #                                                                   dropout_p=self.dropout if self.training else 0,
            #                                                                   is_causal=True)
            #     qkv.append(cross_attention)

            # for 2_layer reflex attention: 1) 5+2+1; 
            q_sa, k_sa, v_sa = q[:, :5, :, :], k[:, :5, :, :], v[:, :5, :, :]
            self_attention = torch.nn.functional.scaled_dot_product_attention(q_sa, k_sa, v_sa, attn_mask=None,
                                                                            dropout_p=self.dropout if self.training else 0,
                                                                            is_causal=True)
            # 2 heads for last previous layer
            q_ca1, k_ca1, v_ca1 = q[:, 5:7, :, :], prevs[1][0][:, 5:7, :, :], prevs[1][1][:, 5:7, :, :]
            cross_attention1 = torch.nn.functional.scaled_dot_product_attention(q_ca1, k_ca1, v_ca1, attn_mask=None,
                                                                          dropout_p=self.dropout if self.training else 0,
                                                                          is_causal=True)
            # 1 head for previous previous layer
            q_ca2, k_ca2, v_ca2 = q[:, 7, :, :], prevs[0][0][:, 7, :, :], prevs[0][1][:, 7, :, :]
            cross_attention2 = torch.nn.functional.scaled_dot_product_attention(q_ca2.unsqueeze(1), k_ca2.unsqueeze(1), v_ca2.unsqueeze(1), attn_mask=None,
                                                                          dropout_p=self.dropout if self.training else 0,
                                                                          is_causal=True)

            # # for 2_layer reflex attention: 2) 3+3+2; 
            # q_sa, k_sa, v_sa = q[:, :3, :, :], k[:, :3, :, :], v[:, :3, :, :]
            # self_attention = torch.nn.functional.scaled_dot_product_attention(q_sa, k_sa, v_sa, attn_mask=None,
            #                                                                 dropout_p=self.dropout if self.training else 0,
            #                                                                 is_causal=True)
            # # 3 heads for last previous layer
            # q_ca1, k_ca1, v_ca1 = q[:, 3:6, :, :], prevs[1][0][:, 3:6, :, :], prevs[1][1][:, 3:6, :, :]
            # cross_attention1 = torch.nn.functional.scaled_dot_product_attention(q_ca1, k_ca1, v_ca1, attn_mask=None,
            #                                                               dropout_p=self.dropout if self.training else 0,
            #                                                               is_causal=True)
            # # 2 head for previous previous layer
            # q_ca2, k_ca2, v_ca2 = q[:, 6:, :, :], prevs[0][0][:, 6:, :, :], prevs[0][1][:, 6:, :, :]
            # cross_attention2 = torch.nn.functional.scaled_dot_product_attention(q_ca2, k_ca2, v_ca2, attn_mask=None,
            #                                                               dropout_p=self.dropout if self.training else 0,
            #                                                               is_causal=True)
            
            y = torch.cat([self_attention, cross_attention1, cross_attention2], dim=1)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, prevs):
        res_attn = self.attn(self.ln_1(x), prevs)
        x = x + res_attn[0]
        x = x + self.mlp(self.ln_2(x))
        return x, res_attn[1]

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        prevs = []

        # # all layers attention
        # for i, block in enumerate(self.transformer.h):
        #     x, prev = block(x, prevs)
        #     if self.config.model_type == 'reflex':
        #         prevs.append(prev)

        # only 2 previous layer attention
        for i, block in enumerate(self.transformer.h):
            x, prev = block(x, prevs)
            if self.config.model_type == 'reflex':
                if i >= 2: prevs.pop(0)
                prevs.append(prev)
                    
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
